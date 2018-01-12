#include "vtkLIC3DRepresentation.h"

#include "vtkAlgorithmOutput.h"
#include "vtkCellData.h"
#include "vtkColorTransferFunction.h"
#include "vtkCommand.h"
#include "vtkCompositeDataToUnstructuredGridFilter.h"
#include "vtkExtentTranslator.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMath.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkPExtentTranslator.h"
#include "vtkPVCacheKeeper.h"
#include "vtkPVLODActor.h"
#include "vtkPVRenderView.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderer.h"
#include "vtkSmartPointer.h"
#include "vtkLIC3DMapper.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStructuredData.h"
#include "vtkTransform.h"
#include "vtkUnsignedCharArray.h"
#include "vtkProjectedTetrahedraMapper.h"
#include "vtkVolume.h"
#include "vtkPVLODVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkResampleToImage.h"
#include "vtkPiecewiseFunction.h"
#include "vtkVolumeRepresentationPreprocessor.h"
#include "vtkSmartVolumeMapper.h"
#include "vtkOutlineSource.h"
#include "vtkOpenGLGPUVolumeRayCastMapper.h"
#include "vtkImageCast.h"

#include <algorithm>
#include <map>
#include <string>

namespace
{
	//----------------------------------------------------------------------------
	void vtkGetNonGhostExtent(int* resultExtent, vtkImageData* dataSet)
	{
		// this is really only meant for topologically structured grids
		dataSet->GetExtent(resultExtent);

		if (vtkUnsignedCharArray* ghostArray = vtkUnsignedCharArray::SafeDownCast(
			dataSet->GetCellData()->GetArray(vtkDataSetAttributes::GhostArrayName())))
		{
			// We have a ghost array. We need to iterate over the array to prune ghost
			// extents.

			int pntExtent[6];
			std::copy(resultExtent, resultExtent + 6, pntExtent);

			int validCellExtent[6];
			vtkStructuredData::GetCellExtentFromPointExtent(pntExtent, validCellExtent);

			// The start extent is the location of the first cell with ghost value 0.
			vtkIdType numTuples = ghostArray->GetNumberOfTuples();
			for (vtkIdType cc = 0; cc < numTuples; ++cc)
			{
				if (ghostArray->GetValue(cc) == 0)
				{
					int ijk[3];
					vtkStructuredData::ComputeCellStructuredCoordsForExtent(cc, pntExtent, ijk);
					validCellExtent[0] = ijk[0];
					validCellExtent[2] = ijk[1];
					validCellExtent[4] = ijk[2];
					break;
				}
			}

			// The end extent is the  location of the last cell with ghost value 0.
			for (vtkIdType cc = (numTuples - 1); cc >= 0; --cc)
			{
				if (ghostArray->GetValue(cc) == 0)
				{
					int ijk[3];
					vtkStructuredData::ComputeCellStructuredCoordsForExtent(cc, pntExtent, ijk);
					validCellExtent[1] = ijk[0];
					validCellExtent[3] = ijk[1];
					validCellExtent[5] = ijk[2];
					break;
				}
			}

			// convert cell-extents to pt extents.
			resultExtent[0] = validCellExtent[0];
			resultExtent[2] = validCellExtent[2];
			resultExtent[4] = validCellExtent[4];

			resultExtent[1] = std::min(validCellExtent[1] + 1, resultExtent[1]);
			resultExtent[3] = std::min(validCellExtent[3] + 1, resultExtent[3]);
			resultExtent[5] = std::min(validCellExtent[5] + 1, resultExtent[5]);
		}
	}
}

vtkStandardNewMacro(vtkLIC3DRepresentation);

vtkLIC3DRepresentation::vtkLIC3DRepresentation()
{
	//this->LICMapper = vtkLIC3DMapper::New();
	//this->Property = vtkProperty::New();
	//
	//this->Actor = vtkPVLODActor::New();
	//this->Actor->SetProperty(this->Property);
	//this->Actor->SetEnableLOD(0);

	this->ResampleToImageFilter = vtkResampleToImage::New();
	this->ResampleToImageFilter->SetSamplingDimensions(64, 64, 64);

	this->Preprocessor = vtkVolumeRepresentationPreprocessor::New();
	this->Preprocessor->SetTetrahedraOnly(1);
	this->RayCastMapper = vtkProjectedTetrahedraMapper::New();
	this->VolumeMapper = vtkLIC3DMapper::New();
	this->Volume = vtkPVLODVolume::New();
	this->VolProperty = vtkVolumeProperty::New();
	this->VolProperty->SetInterpolationType(1);
	this->Volume->SetProperty(this->VolProperty);
	this->Volume->SetMapper(this->VolumeMapper);
	this->Volume->SetEnableLOD(0);

	this->CacheKeeper = vtkPVCacheKeeper::New();

	//this->Cache = vtkImageData::New();

	this->MBMerger = vtkCompositeDataToUnstructuredGridFilter::New();

	//this->CacheKeeper->SetInputData(this->Cache);

	vtkMath::UninitializeBounds(this->DataBounds);
	this->DataSize = 0;

	this->Origin[0] = this->Origin[1] = this->Origin[2] = 0;
	this->Spacing[0] = this->Spacing[1] = this->Spacing[2] = 0;
	this->WholeExtent[0] = this->WholeExtent[2] = this->WholeExtent[4] = 0;
	this->WholeExtent[1] = this->WholeExtent[3] = this->WholeExtent[5] = -1;
	this->OutlineSource = vtkOutlineSource::New();
}


vtkLIC3DRepresentation::~vtkLIC3DRepresentation()
{
	//this->LICMapper->Delete();
	//this->Property->Delete();
	//this->Actor->Delete();
	this->CacheKeeper->Delete();
	//this->Cache->Delete();
	this->MBMerger->Delete();

	this->Preprocessor->Delete();
	this->ResampleToImageFilter->Delete();
	this->RayCastMapper->Delete();
	this->VolumeMapper->Delete();
	this->VolProperty->Delete();
	this->Volume->Delete();

	this->OutlineSource->Delete();
}

int vtkLIC3DRepresentation::FillInputPortInformation(int, vtkInformation* info)
{
	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkUnstructuredGridBase");
	info->Append(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
	info->Append(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkUnstructuredGrid");
	info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
	return 1;
}

int vtkLIC3DRepresentation::ProcessViewRequest(
	vtkInformationRequestKey* request_type, vtkInformation* inInfo, vtkInformation* outInfo)
{
	if (!this->Superclass::ProcessViewRequest(request_type, inInfo, outInfo))
	{
		this->MarkModified();
		return 0;
	}
	if (request_type == vtkPVView::REQUEST_UPDATE())
	{
		//vtkPVRenderView::SetPiece(inInfo, this, this->CacheKeeper->GetOutputDataObject(0));
		vtkPVRenderView::SetPiece(
			inInfo, this, this->OutlineSource->GetOutputDataObject(0), this->DataSize);
		// BUG #14792.
		// We report this->DataSize explicitly since the data being "delivered" is
		// not the data that should be used to make rendering decisions based on
		// data size.
		outInfo->Set(vtkPVRenderView::NEED_ORDERED_COMPOSITING(), 1);
		
		//vtkNew<vtkMatrix4x4> matrix;
		//this->Volume->GetMatrix(matrix.GetPointer());
		vtkPVRenderView::SetGeometryBounds(inInfo, this->DataBounds);

		// Pass partitioning information to the render view.
		vtkPVRenderView::SetOrderedCompositingInformation(inInfo, this,
			this->PExtentTranslator.GetPointer(), this->WholeExtent, this->Origin, this->Spacing);

		vtkPVRenderView::SetRequiresDistributedRendering(inInfo, this, true);
		this->Volume->SetMapper(NULL);
	}
	else if (request_type == vtkPVView::REQUEST_UPDATE_LOD())
	{
		vtkPVRenderView::SetRequiresDistributedRenderingLOD(inInfo, this, true);
	}
	else if (request_type == vtkPVView::REQUEST_RENDER())
	{
		this->UpdateMapperParameters();
	}

	this->MarkModified();

	return 1;
}

int vtkLIC3DRepresentation::RequestData(
	vtkInformation* request, vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
	vtkMath::UninitializeBounds(this->DataBounds);
	this->DataSize = 0;
	this->Origin[0] = this->Origin[1] = this->Origin[2] = 0;
	this->Spacing[0] = this->Spacing[1] = this->Spacing[2] = 0;
	this->WholeExtent[0] = this->WholeExtent[2] = this->WholeExtent[4] = 0;
	this->WholeExtent[1] = this->WholeExtent[3] = this->WholeExtent[5] = -1;

	// Pass caching information to the cache keeper.
	this->CacheKeeper->SetCachingEnabled(this->GetUseCache());
	this->CacheKeeper->SetCacheTime(this->GetCacheKey());

	if (inputVector[0]->GetNumberOfInformationObjects() == 1)
	{
		vtkDataObject* inputDO = vtkDataObject::GetData(inputVector[0], 0);
		this->ResampleToImageFilter->SetInputDataObject(inputDO);
		this->CacheKeeper->SetInputConnection(this->ResampleToImageFilter->GetOutputPort(0));
		this->CacheKeeper->Update();

		this->Volume->SetEnableLOD(0);
		this->VolumeMapper->SetInputConnection(this->CacheKeeper->GetOutputPort());

		//vtkSmartPointer<vtkImageCast> castFilter =
		//	vtkSmartPointer<vtkImageCast>::New();
		//castFilter->SetInputConnection(this->CacheKeeper->GetOutputPort());
		//castFilter->set
		//castFilter->SetOutputScalarTypeToFloat();
		//castFilter->Update();

		vtkImageData* output = vtkImageData::SafeDownCast(this->CacheKeeper->GetOutputDataObject(0));
		this->OutlineSource->SetBounds(output->GetBounds());
		this->OutlineSource->GetBounds(this->DataBounds);
		this->OutlineSource->Update();
		this->DataSize = output->GetActualMemorySize();

		this->PExtentTranslator->GatherExtents(output);
		output->GetOrigin(this->Origin);
		output->GetSpacing(this->Spacing);
		vtkStreamingDemandDrivenPipeline::GetWholeExtent(
			this->CacheKeeper->GetOutputInformation(0), this->WholeExtent);
	}
	else
	{
		// when no input is present, it implies that this processes is on a node
		// without the data input i.e. either client or render-server.
		//this->LICMapper->RemoveAllInputs();
		//this->RayCastMapper->RemoveAllInputs();
		this->VolumeMapper->RemoveAllInputs();
		this->Volume->SetEnableLOD(1);
	}

	return this->Superclass::RequestData(request, inputVector, outputVector);
}

//----------------------------------------------------------------------------
bool vtkLIC3DRepresentation::IsCached(double cache_key)
{
	return this->CacheKeeper->IsCached(cache_key);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::MarkModified()
{
	if (!this->GetUseCache())
	{
		// Cleanup caches when not using cache.
		this->CacheKeeper->RemoveAllCaches();
	}
	this->Superclass::MarkModified();
}

//----------------------------------------------------------------------------
bool vtkLIC3DRepresentation::AddToView(vtkView* view)
{
	// FIXME: Need generic view API to add props.
	vtkPVRenderView* rview = vtkPVRenderView::SafeDownCast(view);
	if (rview)
	{
		//rview->GetRenderer()->AddActor(this->Actor);
		rview->GetRenderer()->AddVolume(this->Volume);
		// Indicate that this is a prop to be rendered during hardware selection.
		return this->Superclass::AddToView(view);
	}
	return false;
}

//----------------------------------------------------------------------------
bool vtkLIC3DRepresentation::RemoveFromView(vtkView* view)
{
	vtkPVRenderView* rview = vtkPVRenderView::SafeDownCast(view);
	if (rview)
	{
		//rview->GetRenderer()->RemoveActor(this->Actor);
		rview->GetRenderer()->RemoveActor(this->Volume);
		return this->Superclass::RemoveFromView(view);
	}
	return false;
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::UpdateMapperParameters()
{
	//this->Actor->SetMapper(this->LICMapper);
	//this->Actor->SetVisibility(1);
	const char* colorArrayName = NULL;
	int fieldAssociation = vtkDataObject::FIELD_ASSOCIATION_POINTS;

	vtkInformation* info = this->GetInputArrayInformation(0);
	if (info && info->Has(vtkDataObject::FIELD_ASSOCIATION()) &&
		info->Has(vtkDataObject::FIELD_NAME()))
	{
		colorArrayName = info->Get(vtkDataObject::FIELD_NAME());
		fieldAssociation = vtkDataObject::FIELD_ASSOCIATION_POINTS;
	}
	//this->RayCastMapper->SelectScalarArray(colorArrayName);
	this->VolumeMapper->SelectScalarArray(colorArrayName);

	switch (fieldAssociation)
	{
	case vtkDataObject::FIELD_ASSOCIATION_CELLS:
		//this->RayCastMapper->SetScalarMode(VTK_SCALAR_MODE_USE_CELL_FIELD_DATA);
		this->VolumeMapper->SetScalarMode(VTK_SCALAR_MODE_USE_CELL_FIELD_DATA);
		//this->LODMapper->SetScalarMode(VTK_SCALAR_MODE_USE_CELL_FIELD_DATA);
		break;

	case vtkDataObject::FIELD_ASSOCIATION_NONE:
		//this->RayCastMapper->SetScalarMode(VTK_SCALAR_MODE_USE_FIELD_DATA);
		this->VolumeMapper->SetScalarMode(VTK_SCALAR_MODE_USE_FIELD_DATA);
		//this->LODMapper->SetScalarMode(VTK_SCALAR_MODE_USE_FIELD_DATA);
		break;

	case vtkDataObject::FIELD_ASSOCIATION_POINTS:
	default:
		//this->RayCastMapper->SetScalarMode(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA);
		this->VolumeMapper->SetScalarMode(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA);
		//this->LODMapper->SetScalarMode(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA);
		break;
	}

	this->Volume->SetMapper(this->VolumeMapper);
	this->Volume->SetVisibility(1);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os, indent);
}

//***************************************************************************
// Forwarded to vtkVolumeProperty.
//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetInterpolationType(int val)
{
	this->VolProperty->SetInterpolationType(val);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetColor(vtkColorTransferFunction* lut)
{
	this->VolProperty->SetColor(lut);
	//this->LODMapper->SetLookupTable(lut);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetScalarOpacity(vtkPiecewiseFunction* pwf)
{
	this->VolProperty->SetScalarOpacity(pwf);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetScalarOpacityUnitDistance(double val)
{
	this->VolProperty->SetScalarOpacityUnitDistance(val);
}

//***************************************************************************
// Forwarded to Property.
//----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetColor(double r, double g, double b)
//{
//	//this->Property->SetColor(r, g, b);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetLineWidth(double val)
//{
//	//this->Property->SetLineWidth(val);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetOpacity(double val)
//{
//	//this->Property->SetOpacity(val);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetPointSize(double val)
//{
//	//this->Property->SetPointSize(val);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetAmbientColor(double r, double g, double b)
//{
//	//this->Property->SetAmbientColor(r, g, b);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetDiffuseColor(double r, double g, double b)
//{
//	//this->Property->SetDiffuseColor(r, g, b);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetEdgeColor(double r, double g, double b)
//{
//	//this->Property->SetEdgeColor(r, g, b);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetInterpolation(int val)
//{
//	//this->Property->SetInterpolation(val);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetSpecularColor(double r, double g, double b)
//{
//	//this->Property->SetSpecularColor(r, g, b);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetSpecularPower(double val)
//{
//	//this->Property->SetSpecularPower(val);
//}

//***************************************************************************
// Forwarded to Actor.
//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetVisibility(bool val)
{
	this->Superclass::SetVisibility(val);
	//this->Actor->SetVisibility(val ? 1 : 0);
	this->Volume->SetVisibility(val ? 1 : 0);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetOrientation(double x, double y, double z)
{
	//this->Actor->SetOrientation(x, y, z);
	this->Volume->SetOrientation(x, y, z);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetOrigin(double x, double y, double z)
{
	//this->Actor->SetOrigin(x, y, z);
	this->Volume->SetOrigin(x, y, z);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetPickable(int val)
{
	//this->Actor->SetPickable(val);
	this->Volume->SetPickable(val);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetPosition(double x, double y, double z)
{
	//this->Actor->SetPosition(x, y, z);
	this->Volume->SetPosition(x, y, z);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetScale(double x, double y, double z)
{
	//this->Actor->SetScale(x, y, z);
	this->Volume->SetScale(x, y, z);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetUserTransform(const double matrix[16])
{
	vtkNew<vtkTransform> transform;
	transform->SetMatrix(matrix);
	//this->Actor->SetUserTransform(transform.GetPointer());
	this->Volume->SetUserTransform(transform.GetPointer());
}

//***************************************************************************
// Forwarded to StreamLinesMapper.
//----------------------------------------------------------------------------
//
void vtkLIC3DRepresentation::SetLICStepSize(double val)
{
	this->VolumeMapper->SetLICStepSize(val);
}

void vtkLIC3DRepresentation::SetNumberOfForwardSteps(int val)
{
	this->VolumeMapper->SetNumberOfForwardSteps(val);
}

void vtkLIC3DRepresentation::SetNumberOfBackwardSteps(int val)
{
	this->VolumeMapper->SetNumberOfBackwardSteps(val);
}

void vtkLIC3DRepresentation::SetGradientScale(double val)
{
	this->VolumeMapper->SetGradientScale(val);
}

void vtkLIC3DRepresentation::SetIllumScale(double val)
{
	this->VolumeMapper->SetIllumScale(val);
}

void vtkLIC3DRepresentation::SetFreqScale(double val)
{
	this->VolumeMapper->SetFreqScale(val);
}

void vtkLIC3DRepresentation::SetVolumeStepScale(double val)
{
	this->VolumeMapper->SetVolumeStepScale(val);
}

//----------------------------------------------------------------------------
void vtkLIC3DRepresentation::SetVolumeDimension(int x, int y, int z)
{
	this->ResampleToImageFilter->SetSamplingDimensions(x, y, z);
	//this->ResampleToImageFilter->Update();
}
//----------------------------------------------------------------------------
const char* vtkLIC3DRepresentation::GetColorArrayName()
{
	vtkInformation* info = this->GetInputArrayInformation(0);
	if (info && info->Has(vtkDataObject::FIELD_ASSOCIATION()) &&
		info->Has(vtkDataObject::FIELD_NAME()))
	{
		return info->Get(vtkDataObject::FIELD_NAME());
	}
	return NULL;
}

//****************************************************************************
// Methods merely forwarding parameters to internal objects.
//****************************************************************************

////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetLookupTable(vtkScalarsToColors* val)
//{
//	this->LICMapper->SetLookupTable(val);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetMapScalars(int val)
//{
//	if (val < 0 || val > 1)
//	{
//		vtkWarningMacro(<< "Invalid parameter for vtkStreamLinesRepresentation::SetMapScalars: "
//			<< val);
//		val = 0;
//	}
//	int mapToColorMode[] = { VTK_COLOR_MODE_DIRECT_SCALARS, VTK_COLOR_MODE_MAP_SCALARS };
//	this->LICMapper->SetColorMode(mapToColorMode[val]);
//}
//
////----------------------------------------------------------------------------
//void vtkLIC3DRepresentation::SetInterpolateScalarsBeforeMapping(int val)
//{
//	this->LICMapper->SetInterpolateScalarsBeforeMapping(val);
//}

void vtkLIC3DRepresentation::SetInputArrayToProcess(
	int idx, int port, int connection, int fieldAssociation, const char* name)
{
	this->Superclass::SetInputArrayToProcess(idx, port, connection, fieldAssociation, name);

	if (idx == 1)
	{
		return;
	}

	//this->LICMapper->SetInputArrayToProcess(idx, port, connection, fieldAssociation, name);
	//this->RayCastMapper->SetInputArrayToProcess(idx, port, connection, fieldAssociation, name);
	this->VolumeMapper->SetInputArrayToProcess(idx, port, connection, fieldAssociation, name);

	//if (name && name[0])
	//{
	//	this->LICMapper->SetScalarVisibility(1);
	//	this->LICMapper->SelectColorArray(name);
	//	this->LICMapper->SetUseLookupTableScalarRange(1);
	//	
	//}
	//else
	//{
	//	this->LICMapper->SetScalarVisibility(0);
	//	this->LICMapper->SelectColorArray(static_cast<const char*>(NULL));
	//}

	switch (fieldAssociation)
	{
	case vtkDataObject::FIELD_ASSOCIATION_CELLS:
		//this->LICMapper->SetScalarMode(VTK_SCALAR_MODE_USE_CELL_FIELD_DATA);
		//this->RayCastMapper->SetScalarMode(VTK_SCALAR_MODE_USE_CELL_FIELD_DATA);
		this->VolumeMapper->SetScalarMode(VTK_SCALAR_MODE_USE_CELL_FIELD_DATA);
		break;

	case vtkDataObject::FIELD_ASSOCIATION_POINTS:
	default:
		//this->LICMapper->SetScalarMode(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA);
		//this->RayCastMapper->SetScalarMode(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA);
		this->VolumeMapper->SetScalarMode(VTK_SCALAR_MODE_USE_POINT_FIELD_DATA);
		break;
	}
}