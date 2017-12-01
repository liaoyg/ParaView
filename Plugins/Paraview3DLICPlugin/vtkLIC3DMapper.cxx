#include "vtkLIC3DMapper.h"

#include "vtkOpenGLVolumeGradientOpacityTable.h"
#include "vtkOpenGLVolumeOpacityTable.h"
#include "vtkOpenGLVolumeRGBTable.h"
#include "vtkOpenGLTransferFunction2D.h"
#include "vtkVolumeShaderComposer.h"
#include "vtkVolumeStateRAII.h"

// Include compiled shader code
#include <raycasterfs.h>
#include <raycastervs.h>

// VTK includes
#include <vtkBoundingBox.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkClipConvexPolyData.h>
#include <vtkColorTransferFunction.h>
#include <vtkCommand.h>
#include <vtkContourFilter.h>
#include <vtkDataArray.h>
#include <vtkDensifyPolyData.h>
#include <vtkFloatArray.h>
#include <vtkOpenGLFramebufferObject.h>
#include <vtkImageData.h>
#include "vtkInformation.h"
#include <vtkLightCollection.h>
#include <vtkLight.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include "vtkOpenGLActor.h"
#include <vtkOpenGLError.h>
#include <vtkOpenGLBufferObject.h>
#include <vtkOpenGLCamera.h>
#include <vtkOpenGLRenderPass.h>
#include <vtkOpenGLRenderUtilities.h>
#include <vtkOpenGLRenderWindow.h>
#include "vtkOpenGLResourceFreeCallback.h"
#include <vtkOpenGLShaderCache.h>
#include <vtkOpenGLVertexArrayObject.h>
#include <vtkPerlinNoise.h>
#include <vtkPixelBufferObject.h>
#include <vtkPixelExtent.h>
#include <vtkPixelTransfer.h>
#include <vtkPlaneCollection.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkShader.h>
#include <vtkShaderProgram.h>
#include <vtkSmartPointer.h>
#include <vtkTessellatedBoxSource.h>
#include <vtkTextureObject.h>
#include <vtkTimerLog.h>
#include <vtkTransform.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkVolumeMask.h>
#include <vtkVolumeProperty.h>
#include <vtkVolumeTexture.h>
#include <vtkWeakPointer.h>
#include <vtkHardwareSelector.h>
#include <vtkImageCast.h>

extern const char* LICraycasterfs;
extern const char* LICcomputefs;
extern const char* LICraycastervs;


vtkStandardNewMacro(vtkLIC3DMapper);

//----------------------------------------------------------------------------
class vtkLIC3DMapper::vtkInternal
{
public:
	// Constructor
	//--------------------------------------------------------------------------
	vtkInternal(vtkLIC3DMapper* parent)
	{
		this->Parent = parent;
		this->ValidTransferFunction = false;
		this->LoadDepthTextureExtensionsSucceeded = false;
		this->CameraWasInsideInLastUpdate = false;
		this->CubeVBOId = 0;
		this->CubeVAOId = 0;
		this->CubeIndicesId = 0;
		this->VolumeTextureObject = nullptr;
		this->NoiseTextureObject = nullptr;
		this->LICNoiseTextureObject = nullptr;
		this->KernelFilterTextureObject = nullptr;
		this->DepthTextureObject = nullptr;
		this->TextureWidth = 1024;
		this->ActualSampleDistance = 1.0;
		this->RGBTables = nullptr;
		this->OpacityTables = nullptr;
		this->Mask1RGBTable = nullptr;
		this->Mask2RGBTable = nullptr;
		this->GradientOpacityTables = nullptr;
		this->TransferFunctions2D = nullptr;
		this->CurrentMask = nullptr;
		this->Dimensions[0] = this->Dimensions[1] = this->Dimensions[2] = -1;
		this->TextureSize[0] = this->TextureSize[1] = this->TextureSize[2] = -1;
		this->WindowLowerLeft[0] = this->WindowLowerLeft[1] = 0;
		this->WindowSize[0] = this->WindowSize[1] = 0;
		this->LastDepthPassWindowSize[0] = this->LastDepthPassWindowSize[1] = 0;
		this->LastRenderToImageWindowSize[0] = 0;
		this->LastRenderToImageWindowSize[1] = 0;
		this->CurrentSelectionPass = vtkHardwareSelector::MIN_KNOWN_PASS - 1;

		this->CellScale[0] = this->CellScale[1] = this->CellScale[2] = 0.0;

		this->UseNoiseGradient = true;
		this->NoiseTextureData = nullptr;
		this->LICNoiseTextureData = nullptr;
		this->LICNoiseGradientData = nullptr;
		this->KernelFilterData = nullptr;

		this->NumberOfLights = 0;
		this->LightComplexity = 0;

		this->Extents[0] = VTK_INT_MAX;
		this->Extents[1] = VTK_INT_MIN;
		this->Extents[2] = VTK_INT_MAX;
		this->Extents[3] = VTK_INT_MIN;
		this->Extents[4] = VTK_INT_MAX;
		this->Extents[5] = VTK_INT_MIN;

		this->CellToPointMatrix->Identity();
		this->AdjustedTexMin[0] = this->AdjustedTexMin[1] = this->AdjustedTexMin[2] = 0.0f;
		this->AdjustedTexMin[3] = 1.0f;
		this->AdjustedTexMax[0] = this->AdjustedTexMax[1] = this->AdjustedTexMax[2] = 1.0f;
		this->AdjustedTexMax[3] = 1.0f;

		this->Scale.clear();
		this->Bias.clear();

		this->NeedToInitializeResources = false;
		this->ShaderCache = nullptr;

		this->FBO = nullptr;
		this->RTTDepthBufferTextureObject = nullptr;
		this->RTTDepthTextureObject = nullptr;
		this->RTTColorTextureObject = nullptr;
		this->RTTDepthTextureType = -1;

		this->DPFBO = nullptr;
		this->DPDepthBufferTextureObject = nullptr;
		this->DPColorTextureObject = nullptr;
		this->PreserveViewport = false;
		this->PreserveGLState = false;
	}

	// Destructor
	//--------------------------------------------------------------------------
	~vtkInternal()
	{
		delete[] this->NoiseTextureData;

		if (this->NoiseTextureObject)
		{
			this->NoiseTextureObject->Delete();
			this->NoiseTextureObject = nullptr;
		}

		if (this->DepthTextureObject)
		{
			this->DepthTextureObject->Delete();
			this->DepthTextureObject = nullptr;
		}

		if (this->FBO)
		{
			this->FBO->Delete();
			this->FBO = nullptr;
		}

		if (this->RTTDepthBufferTextureObject)
		{
			this->RTTDepthBufferTextureObject->Delete();
			this->RTTDepthBufferTextureObject = nullptr;
		}

		if (this->RTTDepthTextureObject)
		{
			this->RTTDepthTextureObject->Delete();
			this->RTTDepthTextureObject = nullptr;
		}

		if (this->RTTColorTextureObject)
		{
			this->RTTColorTextureObject->Delete();
			this->RTTColorTextureObject = nullptr;
		}

		if (this->ImageSampleFBO)
		{
			this->ImageSampleFBO->Delete();
			this->ImageSampleFBO = nullptr;
		}

		for (auto& tex : this->ImageSampleTexture)
		{
			tex = nullptr;
		}
		this->ImageSampleTexture.clear();
		this->ImageSampleTexNames.clear();

		if (this->ImageSampleVBO)
		{
			this->ImageSampleVBO->Delete();
			this->ImageSampleVBO = nullptr;
		}

		if (this->ImageSampleVAO)
		{
			this->ImageSampleVAO->Delete();
			this->ImageSampleVAO = nullptr;
		}
		this->DeleteTransfer1D();
		this->DeleteTransfer2D();

		this->Scale.clear();
		this->Bias.clear();

		// Do not delete the shader programs - Let the cache clean them up.
		this->ImageSampleProg = nullptr;
	}

	// Helper methods
	//--------------------------------------------------------------------------
	template<typename T>
	static void ToFloat(const T& in1, const T& in2, float(&out)[2]);
	template<typename T>
	static void ToFloat(const T& in1, const T& in2, const T& in3,
		float(&out)[3]);
	template<typename T>
	static void ToFloat(T* in, float* out, int noOfComponents);
	template<typename T>
	static void ToFloat(T(&in)[3], float(&out)[3]);
	template<typename T>
	static void ToFloat(T(&in)[2], float(&out)[2]);
	template<typename T>
	static void ToFloat(T& in, float& out);
	template<typename T>
	static void ToFloat(T(&in)[4][2], float(&out)[4][2]);

	///@{
	/**
	* \brief Setup and clean-up 1D and 2D transfer functions.
	*/
	void InitializeTransferFunction(vtkRenderer* ren, vtkVolume* vol,
		int noOfComponents, int independentComponents);

	void UpdateTransferFunction(vtkRenderer* ren, vtkVolume* vol,
		int noOfComponents, int independentComponents);

	void ActivateTransferFunction(vtkShaderProgram* prog, vtkVolumeProperty* volProp,
		int numSamplers);

	void DeactivateTransferFunction(vtkVolumeProperty* volumeProperty,
		int numSamplers);

	void SetupTransferFunction1D(vtkRenderer* ren, int noOfComponents,
		int independentComponents);
	void ReleaseGraphicsTransfer1D(vtkWindow* window);
	void DeleteTransfer1D();

	void SetupTransferFunction2D(vtkRenderer* ren, int noOfComponents,
		int independentComponents);
	void ReleaseGraphicsTransfer2D(vtkWindow* window);
	void DeleteTransfer2D();
	///@}

	bool LoadMask(vtkRenderer* ren, vtkImageData* input, vtkImageData* maskInput,
		vtkVolume* volume);

	bool LoadData(vtkRenderer* ren, vtkVolume* vol, vtkVolumeProperty* volProp,
		vtkImageData* input, vtkDataArray* scalars);

	void ComputeBounds(vtkImageData* input);

	// Update OpenGL volume information
	void UpdateVolume(vtkVolumeProperty* volumeProperty);

	// Update transfer color function based on the incoming inputs
	// and number of scalar components.
	int UpdateColorTransferFunction(vtkRenderer* ren,
		vtkVolume* vol,
		unsigned int component);

	// Scalar opacity
	int UpdateOpacityTransferFunction(vtkRenderer* ren,
		vtkVolume* vol,
		unsigned int component);

	int UpdateGradientOpacityTransferFunction(vtkRenderer* ren,
		vtkVolume* vol,
		unsigned int component);

	void UpdateTransferFunction2D(vtkRenderer* ren, vtkVolume* vol,
		unsigned int component);

	void LoadVectorVolumeTexture(vtkRenderer* ren, vtkImageData* input, vtkDataArray* scalars);

	// Update noise texture (used to reduce rendering artifacts
	// specifically banding effects)
	void CreateNoiseTexture(vtkRenderer* ren);

	// Update noise(random) value for linear convolution
	// White noise is the basic noise model
	void* LoadNoiseData(const char *fileName, int* datasize);
	void* loadGradients(const char *fileName, int* datasize);
	void CreateLICNoiseTexture(vtkRenderer* ren);

	void CreateKernelFilterTexture(vtkRenderer* ren);

	// Update depth texture (used for early termination of the ray)
	void CaptureDepthTexture(vtkRenderer* ren, vtkVolume* vol);

	// Test if camera is inside the volume geometry
	bool IsCameraInside(vtkRenderer* ren, vtkVolume* vol);

	// Compute transformation from cell texture-coordinates to point texture-coords
	// (CTP). Cell data maps correctly to OpenGL cells, point data does not (VTK
	// defines points at the cell corners). To set the point data in the center of the
	// OpenGL texels, a translation of 0.5 texels is applied, and the range is rescaled
	// to the point range.
	//
	// delta = TextureExtentsMax - TextureExtentsMin;
	// min   = vec3(0.5) / delta;
	// max   = (delta - vec3(0.5)) / delta;
	// range = max - min
	//
	// CTP = translation * Scale
	// CTP = range.x,        0,        0,  min.x
	//             0,  range.y,        0,  min.y
	//             0,        0,  range.z,  min.z
	//             0,        0,        0,    1.0
	void ComputeCellToPointMatrix();

	// Update parameters for lighting that will be used in the shader.
	void SetLightingParameters(vtkRenderer* ren,
		vtkShaderProgram* prog,
		vtkVolume* vol);

	// Update the volume geometry
	void RenderVolumeGeometry(vtkRenderer* ren,
		vtkShaderProgram* prog,
		vtkVolume* vol);

	// Update cropping params to shader
	void SetCroppingRegions(vtkRenderer* ren, vtkShaderProgram* prog,
		vtkVolume* vol);

	// Update clipping params to shader
	void SetClippingPlanes(vtkRenderer* ren, vtkShaderProgram* prog,
		vtkVolume* vol);

	// Update the interval of sampling
	void UpdateSamplingDistance(vtkImageData *input,
		vtkRenderer* ren, vtkVolume* vol);

	// Check if the mapper should enter picking mode.
	void CheckPickingState(vtkRenderer* ren);

	// Look for property keys used to control the mapper's state.
	// This is necessary for some render passes which need to ensure
	// a specific OpenGL state when rendering through this mapper.
	void CheckPropertyKeys(vtkVolume* vol);

	// Configure the vtkHardwareSelector to begin a picking pass.
	void BeginPicking(vtkRenderer* ren);

	// Update the prop Id if hardware selection is enabled.
	void SetPickingId(vtkRenderer* ren);

	// Configure the vtkHardwareSelector to end a picking pass.
	void EndPicking(vtkRenderer* ren);

	// Load OpenGL extensiosn required to grab depth sampler buffer
	void LoadRequireDepthTextureExtensions(vtkRenderWindow* renWin);

	// Create GL buffers
	void CreateBufferObjects();

	// Dispose / free GL buffers
	void DeleteBufferObjects();

	// Convert vtkTextureObject to vtkImageData
	void ConvertTextureToImageData(vtkTextureObject* texture,
		vtkImageData* output);

	// Render to texture for final rendering
	void SetupRenderToTexture(vtkRenderer* ren);
	void ExitRenderToTexture(vtkRenderer* ren);

	// Render to texture for depth pass
	void SetupDepthPass(vtkRenderer* ren);
	void ExitDepthPass(vtkRenderer* ren);

	//@{
	/**
	* Image XY-Sampling
	* Render to an internal framebuffer with lower resolution than the currently
	* bound one (hence casting less rays and improving performance). The rendered
	* image is subsequently rendered as a texture-mapped quad (linearly
	* interpolated) to the default (or previously attached) framebuffer. If a
	* vtkOpenGLRenderPass is attached, a variable number of render targets are
	* supported (as specified by the RenderPass). The render targets are assumed
	* to be ordered from GL_COLOR_ATTACHMENT0 to GL_COLOR_ATTACHMENT$N$, where
	* $N$ is the number of targets specified (targets of the previously bound
	* framebuffer as activated through ActivateDrawBuffers(int)). Without a
	* RenderPass attached, it relies on FramebufferObject to re-activate the
	* appropriate previous DrawBuffer.
	*
	* \sa vtkOpenGLRenderPass vtkOpenGLFramebufferObject
	*/
	void BeginImageSample(vtkRenderer* ren, vtkVolume* vol);
	bool InitializeImageSampleFBO(vtkRenderer* ren);
	void EndImageSample(vtkRenderer* ren);
	size_t GetNumImageSampleDrawBuffers(vtkVolume* vol);
	//@}

	void ReleaseRenderToTextureGraphicsResources(vtkWindow* win);
	void ReleaseImageSampleGraphicsResources(vtkWindow* win);
	void ReleaseDepthPassGraphicsResources(vtkWindow* win);

	// Private member variables
	//--------------------------------------------------------------------------
	vtkLIC3DMapper* Parent;

	bool ValidTransferFunction;
	bool LoadDepthTextureExtensionsSucceeded;
	bool CameraWasInsideInLastUpdate;

	GLuint CubeVBOId;
	GLuint CubeVAOId;
	GLuint CubeIndicesId;

	vtkTextureObject* VolumeTextureObject;
	vtkTextureObject* NoiseTextureObject;
	vtkTextureObject* DepthTextureObject;
	vtkTextureObject* LICNoiseTextureObject;
	vtkTextureObject* KernelFilterTextureObject;

	int TextureWidth;

	std::vector<double> Scale;
	std::vector<double> Bias;

	float* NoiseTextureData;
	bool UseNoiseGradient;
	unsigned char* LICNoiseTextureData;
	unsigned char* LICNoiseGradientData;
	unsigned char* KernelFilterData;

	float ActualSampleDistance;

	int LastProjectionParallel;
	int Dimensions[3];
	int TextureSize[3];
	int WindowLowerLeft[2];
	int WindowSize[2];
	int LastDepthPassWindowSize[2];
	int LastRenderToImageWindowSize[2];

	double LoadedBounds[6];
	int Extents[6];
	double DatasetStepSize[3];
	double CellScale[3];
	double CellStep[3];
	double CellSpacing[3];

	int NumberOfLights;
	int LightComplexity;

	std::ostringstream ExtensionsStringStream;

	vtkOpenGLVolumeRGBTables* RGBTables;
	std::map<int, std::string> RGBTablesMap;

	vtkOpenGLVolumeOpacityTables* OpacityTables;
	std::map<int, std::string> OpacityTablesMap;

	vtkOpenGLVolumeRGBTable* Mask1RGBTable;
	vtkOpenGLVolumeRGBTable* Mask2RGBTable;
	vtkOpenGLVolumeGradientOpacityTables* GradientOpacityTables;
	std::map<int, std::string> GradientOpacityTablesMap;

	vtkOpenGLTransferFunctions2D* TransferFunctions2D;
	std::map<int, std::string> TransferFunctions2DMap;
	vtkTimeStamp Transfer2DTime;

	vtkTimeStamp ShaderBuildTime;

	vtkNew<vtkMatrix4x4> TextureToDataSetMat;
	vtkNew<vtkMatrix4x4> InverseTextureToDataSetMat;

	vtkNew<vtkMatrix4x4> InverseProjectionMat;
	vtkNew<vtkMatrix4x4> InverseModelViewMat;
	vtkNew<vtkMatrix4x4> InverseVolumeMat;

	vtkNew<vtkMatrix4x4> TextureToEyeTransposeInverse;

	vtkNew<vtkMatrix4x4> TempMatrix1;

	vtkNew<vtkMatrix4x4> CellToPointMatrix;
	float AdjustedTexMin[4];
	float AdjustedTexMax[4];

	vtkSmartPointer<vtkPolyData> BBoxPolyData;

	vtkSmartPointer<vtkVolumeTexture> CurrentMask;

	vtkTimeStamp InitializationTime;
	vtkTimeStamp InputUpdateTime;
	vtkTimeStamp VolumeUpdateTime;
	vtkTimeStamp MaskUpdateTime;
	vtkTimeStamp ReleaseResourcesTime;
	vtkTimeStamp DepthPassTime;
	vtkTimeStamp DepthPassSetupTime;
	vtkTimeStamp SelectionStateTime;
	int CurrentSelectionPass;
	bool IsPicking;

	bool NeedToInitializeResources;
	bool PreserveViewport;
	bool PreserveGLState;

	vtkShaderProgram* ShaderProgram;
	vtkOpenGLShaderCache* ShaderCache;

	vtkOpenGLFramebufferObject* FBO;
	vtkTextureObject* RTTDepthBufferTextureObject;
	vtkTextureObject* RTTDepthTextureObject;
	vtkTextureObject* RTTColorTextureObject;
	int RTTDepthTextureType;

	vtkOpenGLFramebufferObject* DPFBO;
	vtkTextureObject* DPDepthBufferTextureObject;
	vtkTextureObject* DPColorTextureObject;

	vtkOpenGLFramebufferObject* ImageSampleFBO = nullptr;
	std::vector<vtkSmartPointer<vtkTextureObject>> ImageSampleTexture;
	std::vector<std::string> ImageSampleTexNames;
	vtkShaderProgram* ImageSampleProg = nullptr;
	vtkOpenGLVertexArrayObject* ImageSampleVAO = nullptr;
	vtkOpenGLBufferObject* ImageSampleVBO = nullptr;
	size_t NumImageSampleDrawBuffers = 0;
	bool RebuildImageSampleProg = false;
	bool RenderPassAttached = false;

	vtkNew<vtkContourFilter>  ContourFilter;
	vtkNew<vtkPolyDataMapper> ContourMapper;
	vtkNew<vtkActor> ContourActor;
};

//----------------------------------------------------------------------------
template<typename T>
void vtkLIC3DMapper::vtkInternal::ToFloat(
	const T& in1, const T& in2, float(&out)[2])
{
	out[0] = static_cast<float>(in1);
	out[1] = static_cast<float>(in2);
}

template<typename T>
void vtkLIC3DMapper::vtkInternal::ToFloat(
	const T& in1, const T& in2, const T& in3, float(&out)[3])
{
	out[0] = static_cast<float>(in1);
	out[1] = static_cast<float>(in2);
	out[2] = static_cast<float>(in3);
}

//----------------------------------------------------------------------------
template<typename T>
void vtkLIC3DMapper::vtkInternal::ToFloat(
	T* in, float* out, int noOfComponents)
{
	for (int i = 0; i < noOfComponents; ++i)
	{
		out[i] = static_cast<float>(in[i]);
	}
}

//----------------------------------------------------------------------------
template<typename T>
void vtkLIC3DMapper::vtkInternal::ToFloat(
	T(&in)[3], float(&out)[3])
{
	out[0] = static_cast<float>(in[0]);
	out[1] = static_cast<float>(in[1]);
	out[2] = static_cast<float>(in[2]);
}

//----------------------------------------------------------------------------
template<typename T>
void vtkLIC3DMapper::vtkInternal::ToFloat(
	T(&in)[2], float(&out)[2])
{
	out[0] = static_cast<float>(in[0]);
	out[1] = static_cast<float>(in[1]);
}

//----------------------------------------------------------------------------
template<typename T>
void vtkLIC3DMapper::vtkInternal::ToFloat(
	T& in, float& out)
{
	out = static_cast<float>(in);
}

//----------------------------------------------------------------------------
template<typename T>
void vtkLIC3DMapper::vtkInternal::ToFloat(
	T(&in)[4][2], float(&out)[4][2])
{
	out[0][0] = static_cast<float>(in[0][0]);
	out[0][1] = static_cast<float>(in[0][1]);
	out[1][0] = static_cast<float>(in[1][0]);
	out[1][1] = static_cast<float>(in[1][1]);
	out[2][0] = static_cast<float>(in[2][0]);
	out[2][1] = static_cast<float>(in[2][1]);
	out[3][0] = static_cast<float>(in[3][0]);
	out[3][1] = static_cast<float>(in[3][1]);
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::SetupTransferFunction1D(
	vtkRenderer* ren, int noOfComponents, int independentComponents)
{
	this->ReleaseGraphicsTransfer1D(ren->GetRenderWindow());
	this->DeleteTransfer1D();

	// Check component mode (independent or dependent)
	noOfComponents = noOfComponents > 1 && independentComponents ?
		noOfComponents : 1;

	// Create RGB and opacity (scalar and gradient) lookup tables. We support up
	// to four components in independentComponents mode.
	this->RGBTables = new vtkOpenGLVolumeRGBTables(noOfComponents);
	this->OpacityTables = new vtkOpenGLVolumeOpacityTables(noOfComponents);
	this->GradientOpacityTables = new vtkOpenGLVolumeGradientOpacityTables(
		noOfComponents);

	this->OpacityTablesMap.clear();
	this->RGBTablesMap.clear();
	this->GradientOpacityTablesMap.clear();

	if (this->Parent->MaskInput != nullptr &&
		this->Parent->MaskType == LabelMapMaskType)
	{
		if (this->Mask1RGBTable == nullptr)
		{
			this->Mask1RGBTable = vtkOpenGLVolumeRGBTable::New();
		}
		if (this->Mask2RGBTable == nullptr)
		{
			this->Mask2RGBTable = vtkOpenGLVolumeRGBTable::New();
		}
	}

	std::ostringstream numeric;
	for (int i = 0; i < noOfComponents; ++i)
	{
		numeric << i;
		if (i > 0)
		{
			this->OpacityTablesMap[i] = std::string("in_opacityTransferFunc") +
				numeric.str();
			this->RGBTablesMap[i] = std::string("in_colorTransferFunc") +
				numeric.str();
			this->GradientOpacityTablesMap[i] = std::string("in_gradientTransferFunc") +
				numeric.str();
		}
		else
		{
			this->OpacityTablesMap[i] = std::string("in_opacityTransferFunc");
			this->RGBTablesMap[i] = std::string("in_colorTransferFunc");
			this->GradientOpacityTablesMap[i] = std::string("in_gradientTransferFunc");
		}
		numeric.str("");
		numeric.clear();
	}

	this->InitializationTime.Modified();
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::SetupTransferFunction2D(
	vtkRenderer* ren, int noOfComponents, int independentComponents)
{
	this->ReleaseGraphicsTransfer2D(ren->GetRenderWindow());
	this->DeleteTransfer2D();

	unsigned int const num = (noOfComponents > 1 && independentComponents) ?
		noOfComponents : 1;
	this->TransferFunctions2D = new vtkOpenGLTransferFunctions2D(num);

	std::ostringstream indexStream;
	const std::string baseName = "in_transfer2D";
	for (unsigned int i = 0; i < num; i++)
	{
		if (i > 0)
		{
			indexStream << i;
		}
		this->TransferFunctions2DMap[0] = baseName + indexStream.str();
		indexStream.str("");
		indexStream.clear();
	}

	this->Transfer2DTime.Modified();
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::InitializeTransferFunction(
	vtkRenderer* ren, vtkVolume* vol, int noOfComponents,
	int independentComponents)
{
	const int transferMode = vol->GetProperty()->GetTransferFunctionMode();
	switch (transferMode)
	{
	case vtkVolumeProperty::TF_2D:
		this->SetupTransferFunction2D(ren, noOfComponents, independentComponents);
		break;

	case vtkVolumeProperty::TF_1D:
	default:
		this->SetupTransferFunction1D(ren, noOfComponents, independentComponents);
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::UpdateTransferFunction(
	vtkRenderer* ren, vtkVolume* vol, int noOfComponents,
	int independentComponents)
{
	int const transferMode = vol->GetProperty()->GetTransferFunctionMode();
	switch (transferMode)
	{
	case vtkVolumeProperty::TF_1D:
		if (independentComponents)
		{
			for (int i = 0; i < noOfComponents; ++i)
			{
				this->UpdateOpacityTransferFunction(ren, vol, i);
				this->UpdateGradientOpacityTransferFunction(ren, vol, i);
				this->UpdateColorTransferFunction(ren, vol, i);
			}
		}
		else
		{
			if (noOfComponents == 2 || noOfComponents == 4)
			{
				this->UpdateOpacityTransferFunction(ren, vol, noOfComponents - 1);
				this->UpdateGradientOpacityTransferFunction(ren, vol,
					noOfComponents - 1);
				this->UpdateColorTransferFunction(ren, vol, 0);
			}
		}
		break;

	case vtkVolumeProperty::TF_2D:
		if (independentComponents)
		{
			for (int i = 0; i < noOfComponents; ++i)
			{
				this->UpdateTransferFunction2D(ren, vol, i);
			}
		}
		else
		{
			if (noOfComponents == 2 || noOfComponents == 4)
			{
				this->UpdateTransferFunction2D(ren, vol, 0);
			}
		}
		break;
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::ActivateTransferFunction(
	vtkShaderProgram* prog, vtkVolumeProperty* volumeProperty,
	int numberOfSamplers)
{
	int const transferMode = volumeProperty->GetTransferFunctionMode();
	switch (transferMode)
	{
	case vtkVolumeProperty::TF_1D:
		for (int i = 0; i < numberOfSamplers; ++i)
		{
			this->OpacityTables->GetTable(i)->Activate();
			prog->SetUniformi(
				this->OpacityTablesMap[i].c_str(),
				this->OpacityTables->GetTable(i)->GetTextureUnit());

			if (this->Parent->BlendMode != vtkGPUVolumeRayCastMapper::ADDITIVE_BLEND)
			{
				this->RGBTables->GetTable(i)->Activate();
				prog->SetUniformi(
					this->RGBTablesMap[i].c_str(),
					this->RGBTables->GetTable(i)->GetTextureUnit());
			}

			if (this->GradientOpacityTables)
			{
				this->GradientOpacityTables->GetTable(i)->Activate();
				prog->SetUniformi(
					this->GradientOpacityTablesMap[i].c_str(),
					this->GradientOpacityTables->GetTable(i)->GetTextureUnit());
			}
		}
		break;
	case vtkVolumeProperty::TF_2D:
		for (int i = 0; i < numberOfSamplers; ++i)
		{
			vtkOpenGLTransferFunction2D* table =
				this->TransferFunctions2D->GetTable(i);
			table->Activate();
			prog->SetUniformi(this->TransferFunctions2DMap[i].c_str(),
				table->GetTextureUnit());
		}
		break;
	}
}

//-----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::DeactivateTransferFunction(
	vtkVolumeProperty* volumeProperty, int numberOfSamplers)
{
	int const transferMode = volumeProperty->GetTransferFunctionMode();
	switch (transferMode)
	{
	case vtkVolumeProperty::TF_1D:
		for (int i = 0; i < numberOfSamplers; ++i)
		{
			this->OpacityTables->GetTable(i)->Deactivate();
			if (this->Parent->BlendMode != vtkGPUVolumeRayCastMapper::ADDITIVE_BLEND)
			{
				this->RGBTables->GetTable(i)->Deactivate();
			}
			if (this->GradientOpacityTables)
			{
				this->GradientOpacityTables->GetTable(i)->Deactivate();
			}
		}
		break;
	case vtkVolumeProperty::TF_2D:
		for (int i = 0; i < numberOfSamplers; ++i)
		{
			this->TransferFunctions2D->GetTable(i)->Deactivate();
		}
		break;
	}
}

//-----------------------------------------------------------------------------
bool vtkLIC3DMapper::vtkInternal::LoadMask(vtkRenderer* ren,
	vtkImageData* vtkNotUsed(input), vtkImageData* maskInput,
	vtkVolume* vtkNotUsed(volume))
{
	bool result = true;
	if (maskInput &&
		(maskInput->GetMTime() > this->MaskUpdateTime))
	{
		if (!this->CurrentMask)
		{
			this->CurrentMask = vtkSmartPointer<vtkVolumeTexture>::New();
			this->CurrentMask->SetMapper(this->Parent);

			const auto& part = this->Parent->VolumeTexture->GetPartitions();
			this->CurrentMask->SetPartitions(part[0], part[1], part[2]);
		}

		vtkDataArray* arr = this->Parent->GetScalars(maskInput,
			this->Parent->ScalarMode, this->Parent->ArrayAccessMode,
			this->Parent->ArrayId, this->Parent->ArrayName, this->Parent->CellFlag);

		result = this->CurrentMask->LoadVolume(ren, maskInput, arr,
			VTK_NEAREST_INTERPOLATION);

		this->MaskUpdateTime.Modified();
	}

	return result;
}

//----------------------------------------------------------------------------
bool vtkLIC3DMapper::vtkInternal::LoadData(vtkRenderer* ren,
	vtkVolume* vol, vtkVolumeProperty* volProp, vtkImageData* input,
	vtkDataArray* scalars)
{
	// Update bounds, data, and geometry
	input->GetDimensions(this->Dimensions);
	bool success = this->Parent->VolumeTexture->LoadVolume(ren, input, scalars,
		volProp->GetInterpolationType());

	this->ComputeBounds(input);
	this->ComputeCellToPointMatrix();
	this->LoadMask(ren, input, this->Parent->MaskInput, vol);
	this->InputUpdateTime.Modified();

	return success;
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::ReleaseGraphicsTransfer1D(
	vtkWindow* window)
{
	if (this->RGBTables)
	{
		this->RGBTables->ReleaseGraphicsResources(window);
	}

	if (this->Mask1RGBTable)
	{
		this->Mask1RGBTable->ReleaseGraphicsResources(window);
	}

	if (this->Mask2RGBTable)
	{
		this->Mask2RGBTable->ReleaseGraphicsResources(window);
	}

	if (this->OpacityTables)
	{
		this->OpacityTables->ReleaseGraphicsResources(window);
	}

	if (this->GradientOpacityTables)
	{
		this->GradientOpacityTables->ReleaseGraphicsResources(window);
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::DeleteTransfer1D()
{
	delete this->RGBTables;
	this->RGBTables = nullptr;

	if (this->Mask1RGBTable)
	{
		this->Mask1RGBTable->Delete();
		this->Mask1RGBTable = nullptr;
	}

	if (this->Mask2RGBTable)
	{
		this->Mask2RGBTable->Delete();
		this->Mask2RGBTable = nullptr;
	}

	delete this->OpacityTables;
	this->OpacityTables = nullptr;

	delete this->GradientOpacityTables;
	this->GradientOpacityTables = nullptr;
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::ReleaseGraphicsTransfer2D(
	vtkWindow* window)
{
	if (this->TransferFunctions2D)
	{
		this->TransferFunctions2D->ReleaseGraphicsResources(window);
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::DeleteTransfer2D()
{
	delete this->TransferFunctions2D;
	this->TransferFunctions2D = nullptr;
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::ComputeBounds(
	vtkImageData* input)
{
	double origin[3];

	input->GetSpacing(this->CellSpacing);
	input->GetOrigin(origin);
	input->GetExtent(this->Extents);

	int swapBounds[3];
	swapBounds[0] = (this->CellSpacing[0] < 0);
	swapBounds[1] = (this->CellSpacing[1] < 0);
	swapBounds[2] = (this->CellSpacing[2] < 0);

	// Loaded data represents points
	if (!this->Parent->CellFlag)
	{
		// If spacing is negative, we may have to rethink the equation
		// between real point and texture coordinate...
		this->LoadedBounds[0] = origin[0] +
			static_cast<double>(this->Extents[0 + swapBounds[0]]) *
			this->CellSpacing[0];
		this->LoadedBounds[2] = origin[1] +
			static_cast<double>(this->Extents[2 + swapBounds[1]]) *
			this->CellSpacing[1];
		this->LoadedBounds[4] = origin[2] +
			static_cast<double>(this->Extents[4 + swapBounds[2]]) *
			this->CellSpacing[2];
		this->LoadedBounds[1] = origin[0] +
			static_cast<double>(this->Extents[1 - swapBounds[0]]) *
			this->CellSpacing[0];
		this->LoadedBounds[3] = origin[1] +
			static_cast<double>(this->Extents[3 - swapBounds[1]]) *
			this->CellSpacing[1];
		this->LoadedBounds[5] = origin[2] +
			static_cast<double>(this->Extents[5 - swapBounds[2]]) *
			this->CellSpacing[2];
	}
	// Loaded extents represent cells
	else
	{
		int i = 0;
		while (i < 3)
		{
			this->LoadedBounds[2 * i + swapBounds[i]] = origin[i] +
				(static_cast<double>(this->Extents[2 * i])) *
				this->CellSpacing[i];

			this->LoadedBounds[2 * i + 1 - swapBounds[i]] = origin[i] +
				(static_cast<double>(this->Extents[2 * i + 1]) + 1.0) *
				this->CellSpacing[i];

			i++;
		}
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::UpdateVolume(
	vtkVolumeProperty* volumeProperty)
{
	if (volumeProperty->GetMTime() > this->VolumeUpdateTime.GetMTime())
	{
		int const newInterp = volumeProperty->GetInterpolationType();
		this->Parent->VolumeTexture->UpdateInterpolationType(newInterp);
	}

	this->VolumeUpdateTime.Modified();
}

//----------------------------------------------------------------------------
int vtkLIC3DMapper::vtkInternal::
UpdateColorTransferFunction(vtkRenderer* ren, vtkVolume* vol,
	unsigned int component)
{
	// Volume property cannot be null.
	vtkVolumeProperty* volumeProperty = vol->GetProperty();

	// Build the colormap in a 1D texture.
	// 1D RGB-texture=mapping from scalar values to color values
	// build the table.
	vtkColorTransferFunction* colorTransferFunction =
		volumeProperty->GetRGBTransferFunction(component);

	double componentRange[2];
	for (int i = 0; i < 2; ++i)
	{
		componentRange[i] = this->Parent->VolumeTexture->ScalarRange[component][i];
	}

	// Add points only if its not being added before
	if (colorTransferFunction->GetSize() < 1)
	{
		colorTransferFunction->AddRGBPoint(componentRange[0], 0.0, 0.0, 0.0);
		colorTransferFunction->AddRGBPoint(componentRange[1], 1.0, 1.0, 1.0);
	}

	int filterVal =
		volumeProperty->GetInterpolationType() == VTK_LINEAR_INTERPOLATION ?
		vtkTextureObject::Linear : vtkTextureObject::Nearest;

	this->RGBTables->GetTable(component)->Update(
		volumeProperty->GetRGBTransferFunction(component),
		componentRange,
#if GL_ES_VERSION_3_0 != 1
		filterVal,
#else
		vtkTextureObject::Nearest,
#endif
		vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));

	if (this->Parent->MaskInput != nullptr &&
		this->Parent->MaskType == LabelMapMaskType)
	{
		vtkColorTransferFunction* colorTransferFunc =
			volumeProperty->GetRGBTransferFunction(1);
		this->Mask1RGBTable->Update(colorTransferFunc, componentRange,
			vtkTextureObject::Nearest,
			vtkOpenGLRenderWindow::SafeDownCast(
				ren->GetRenderWindow()));

		colorTransferFunc = volumeProperty->GetRGBTransferFunction(2);
		this->Mask2RGBTable->Update(colorTransferFunc, componentRange,
			vtkTextureObject::Nearest,
			vtkOpenGLRenderWindow::SafeDownCast(
				ren->GetRenderWindow()));
	}

	return 0;
}

//----------------------------------------------------------------------------
int vtkLIC3DMapper::vtkInternal::
UpdateOpacityTransferFunction(vtkRenderer* ren, vtkVolume* vol,
	unsigned int component)
{
	vtkVolumeProperty* volumeProperty = vol->GetProperty();

	// Transfer function table index based on whether independent / dependent
	// components. If dependent, use the first scalar opacity transfer function
	unsigned int lookupTableIndex = volumeProperty->GetIndependentComponents() ?
		component : 0;
	vtkPiecewiseFunction* scalarOpacity =
		volumeProperty->GetScalarOpacity(lookupTableIndex);

	double componentRange[2];
	for (int i = 0; i < 2; ++i)
	{
		componentRange[i] = this->Parent->VolumeTexture->ScalarRange[component][i];
	}

	if (scalarOpacity->GetSize() < 1)
	{
		scalarOpacity->AddPoint(componentRange[0], 0.0);
		scalarOpacity->AddPoint(componentRange[1], 0.5);
	}

	int filterVal =
		volumeProperty->GetInterpolationType() == VTK_LINEAR_INTERPOLATION ?
		vtkTextureObject::Linear : vtkTextureObject::Nearest;

	this->OpacityTables->GetTable(lookupTableIndex)->Update(
		scalarOpacity, this->Parent->BlendMode,
		this->ActualSampleDistance,
		componentRange,
		volumeProperty->GetScalarOpacityUnitDistance(component),
#if GL_ES_VERSION_3_0 != 1
		filterVal,
#else
		vtkTextureObject::Nearest,
#endif
		vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));

	return 0;
}

//------------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::
UpdateTransferFunction2D(vtkRenderer* ren, vtkVolume* vol,
	unsigned int component)
{
	vtkVolumeProperty* prop = vol->GetProperty();
	int const transferMode = prop->GetTransferFunctionMode();
	if (transferMode != vtkVolumeProperty::TF_2D)
	{
		return;
	}

	// Use the first LUT when using dependent components
	unsigned int const lutIndex = prop->GetIndependentComponents() ?
		component : 0;

	vtkImageData* transfer2D = prop->GetTransferFunction2D(lutIndex);
#if GL_ES_VERSION_3_0 != 1
	int const interp = prop->GetInterpolationType() == VTK_LINEAR_INTERPOLATION ?
		vtkTextureObject::Linear : vtkTextureObject::Nearest;
#else
	int const interp = vtkTextureObject::Nearest;
#endif

	this->TransferFunctions2D->GetTable(lutIndex)->Update(transfer2D, interp,
		vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));
}

//----------------------------------------------------------------------------
int vtkLIC3DMapper::vtkInternal::
UpdateGradientOpacityTransferFunction(vtkRenderer* ren, vtkVolume* vol,
	unsigned int component)
{
	vtkVolumeProperty* volumeProperty = vol->GetProperty();

	// Transfer function table index based on whether independent / dependent
	// components. If dependent, use the first gradient opacity transfer function
	unsigned int lookupTableIndex = volumeProperty->GetIndependentComponents() ?
		component : 0;

	if (!volumeProperty->HasGradientOpacity(lookupTableIndex) ||
		!this->GradientOpacityTables)
	{
		return 1;
	}

	vtkPiecewiseFunction* gradientOpacity =
		volumeProperty->GetGradientOpacity(lookupTableIndex);

	double componentRange[2];
	for (int i = 0; i < 2; ++i)
	{
		componentRange[i] = this->Parent->VolumeTexture->ScalarRange[component][i];
	}

	if (gradientOpacity->GetSize() < 1)
	{
		gradientOpacity->AddPoint(componentRange[0], 0.0);
		gradientOpacity->AddPoint(componentRange[1], 0.5);
	}

	int filterVal =
		volumeProperty->GetInterpolationType() == VTK_LINEAR_INTERPOLATION ?
		vtkTextureObject::Linear : vtkTextureObject::Nearest;

	this->GradientOpacityTables->GetTable(lookupTableIndex)->Update(
		gradientOpacity,
		this->ActualSampleDistance,
		componentRange,
		volumeProperty->GetScalarOpacityUnitDistance(component),
#if GL_ES_VERSION_3_0 != 1
		filterVal,
#else
		vtkTextureObject::Nearest,
#endif
		vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));

	return 0;
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::LoadVectorVolumeTexture(vtkRenderer* ren, vtkImageData* input, vtkDataArray* scalars)
{
	vtkOpenGLRenderWindow* glWindow = vtkOpenGLRenderWindow::SafeDownCast(
		ren->GetRenderWindow());
	if (!this->Parent->VectorVolumeTex)
	{
		this->Parent->VectorVolumeTex = vtkTextureObject::New();
	}
	this->Parent->VectorVolumeTex->SetContext(glWindow);
	int size[3];
	int* extent = input->GetExtent();
	for (int i = 0; i < 3; i++)
	{
		size[i] = extent[2 * i + 1] - extent[2 * i] + 1;
	}
	this->Parent->VectorVolumeTex->SetInternalFormat(GL_RGB32F);
	this->Parent->VectorVolumeTex->SetFormat(GL_RGB);
	void* dataPtr = scalars->GetVoidPointer(0);
	// Prepare texture
	bool success = this->Parent->VectorVolumeTex->Create3DFromRaw(size[0], size[1], size[2], 3, VTK_FLOAT,
		dataPtr);

	this->Parent->VectorVolumeTex->SetWrapS(vtkTextureObject::Repeat);
	this->Parent->VectorVolumeTex->SetWrapT(vtkTextureObject::Repeat);
	this->Parent->VectorVolumeTex->SetWrapR(vtkTextureObject::Repeat);
	this->Parent->VectorVolumeTex->SetMagnificationFilter(vtkTextureObject::Linear);
	this->Parent->VectorVolumeTex->SetMinificationFilter(vtkTextureObject::Linear);
	this->Parent->VectorVolumeTex->SetBorderColor(0.0f, 0.0f, 0.0f, 0.0f);
	this->Parent->VectorVolumeTex->Modified();
}


//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::CreateNoiseTexture(
	vtkRenderer* ren)
{
	vtkOpenGLRenderWindow* glWindow = vtkOpenGLRenderWindow::SafeDownCast(
		ren->GetRenderWindow());

	if (!this->NoiseTextureObject)
	{
		this->NoiseTextureObject = vtkTextureObject::New();
	}
	this->NoiseTextureObject->SetContext(glWindow);

	bool updateSize = false;
	bool useUserSize = this->Parent->NoiseTextureSize[0] > 0 &&
		this->Parent->NoiseTextureSize[1] > 0;
	if (useUserSize)
	{
		int const twidth = this->NoiseTextureObject->GetWidth();
		int const theight = this->NoiseTextureObject->GetHeight();
		updateSize = this->Parent->NoiseTextureSize[0] != twidth ||
			this->Parent->NoiseTextureSize[1] != theight;
	}

	if (!this->NoiseTextureObject->GetHandle() || updateSize ||
		this->NoiseTextureObject->GetMTime() < this->Parent->NoiseGenerator->GetMTime())
	{
		int* winSize = ren->GetRenderWindow()->GetSize();
		int sizeX = useUserSize ? this->Parent->NoiseTextureSize[0] : winSize[0];
		int sizeY = useUserSize ? this->Parent->NoiseTextureSize[1] : winSize[1];

		int const maxSize = vtkTextureObject::GetMaximumTextureSize(glWindow);
		if (sizeX > maxSize || sizeY > maxSize)
		{
			sizeX = vtkMath::Max(sizeX, maxSize);
			sizeY = vtkMath::Max(sizeY, maxSize);
		}

		// Allocate buffer. After controlling for the maximum supported size sizeX/Y
		// might have changed, so an additional check is needed.
		int const twidth = this->NoiseTextureObject->GetWidth();
		int const theight = this->NoiseTextureObject->GetHeight();
		bool sizeChanged = sizeX != twidth || sizeY != theight;
		if (sizeChanged || !this->NoiseTextureData)
		{
			delete[] this->NoiseTextureData;
			this->NoiseTextureData = nullptr;
			this->NoiseTextureData = new float[sizeX * sizeY];
		}

		// Generate jitter noise
		if (!this->Parent->NoiseGenerator)
		{
			// Use default settings
			vtkPerlinNoise* perlinNoise = vtkPerlinNoise::New();
			perlinNoise->SetPhase(0.0, 0.0, 0.0);
			perlinNoise->SetFrequency(sizeX, sizeY, 1.0);
			perlinNoise->SetAmplitude(0.5); /* [-n, n] */
			this->Parent->NoiseGenerator = perlinNoise;
		}

		int const bufferSize = sizeX * sizeY;
		for (int i = 0; i < bufferSize; i++)
		{
			int const x = i % sizeX;
			int const y = i / sizeY;
			this->NoiseTextureData[i] = static_cast<float>(
				this->Parent->NoiseGenerator->EvaluateFunction(x, y, 0.0) + 0.1);
		}

		// Prepare texture
		this->NoiseTextureObject->Create2DFromRaw(sizeX, sizeY, 1, VTK_FLOAT,
			this->NoiseTextureData);

		this->NoiseTextureObject->SetWrapS(vtkTextureObject::Repeat);
		this->NoiseTextureObject->SetWrapT(vtkTextureObject::Repeat);
		this->NoiseTextureObject->SetMagnificationFilter(vtkTextureObject::Nearest);
		this->NoiseTextureObject->SetMinificationFilter(vtkTextureObject::Nearest);
		this->NoiseTextureObject->SetBorderColor(0.0f, 0.0f, 0.0f, 0.0f);
		this->NoiseTextureObject->Modified();
	}
}

//----------------------------------------------------------------------------
void* vtkLIC3DMapper::vtkInternal::LoadNoiseData(const char *fileName, int* noisetexturesize)
{
	int datasize[3];
	int size = 0;
	void* rawdata = NULL;
	if (!fileName)
		return NULL;

	//Load Raw data from file
	FILE *src = NULL;
	// try to open the noise file
	src = fopen(fileName, "rb");
	if (!src)
	{
		fprintf(stderr, "NoiseData:  Could not load noise from "
			"(\"%s\").\n", fileName);
		return NULL;
	}

	// read "header"
	if (fread((void*)datasize, 4, 3, src) != 3)
	{
		fprintf(stderr, "NoiseData:  Could not read noise header from "
			"\"%s\".\n", fileName);
		fclose(src);
		return NULL;
	}

	size = datasize[0] * datasize[1] * datasize[2];
	rawdata = new unsigned char[size];
	if (fread(rawdata, datasize[0] * datasize[1],
		datasize[2], src) != static_cast<unsigned int>(datasize[2]))
	{
		fprintf(stderr, "NoiseData:  Error reading noise data from "
			"\"%s\".\n", fileName);
		fclose(src);
		delete[] static_cast<unsigned char*>(rawdata);
		rawdata = NULL;
		return NULL;
	}
	noisetexturesize[0] = datasize[0];
	noisetexturesize[1] = datasize[1];
	noisetexturesize[2] = datasize[2];

	return rawdata;
}

void* vtkLIC3DMapper::vtkInternal::loadGradients(const char *fileName, int* datasize)
{
	int size;
	std::ifstream in;
	void *gradients = NULL;
	
	if (!fileName)
		return NULL;

	in.open(fileName, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		fprintf(stderr, "loadGradients: No pre-computed gradients found.\n");
		return NULL;
	}

	size = 3 * datasize[0] * datasize[1] * datasize[2];
	gradients = new unsigned char[size];
	in.read((char*)gradients, static_cast<std::streamsize>(size));

	if (in.fail())
	{
		fprintf(stderr, "loadGradients: Reading gradients from \"%s\" failed.\n",
			fileName);
		in.close();
		return NULL;
	}
	in.close();

	return gradients;
}

void vtkLIC3DMapper::vtkInternal::CreateLICNoiseTexture(vtkRenderer* ren)
{
	vtkOpenGLRenderWindow* glWindow = vtkOpenGLRenderWindow::SafeDownCast(
		ren->GetRenderWindow());

	if (!this->LICNoiseTextureObject)
	{
		this->LICNoiseTextureObject = vtkTextureObject::New();
	}
	this->LICNoiseTextureObject->SetContext(glWindow);

	bool updateSize = false;
	bool useUserSize = this->Parent->LICNoiseSize[0] > 0 &&
		this->Parent->LICNoiseSize[1] > 0 && this->Parent->LICNoiseSize[2] > 0;
	if (useUserSize)
	{
		int const twidth = this->LICNoiseTextureObject->GetWidth();
		int const theight = this->LICNoiseTextureObject->GetHeight();
		int const tdepth = this->LICNoiseTextureObject->GetDepth();
		updateSize = this->Parent->LICNoiseSize[0] != twidth ||
			this->Parent->LICNoiseSize[1] != theight || this->Parent->LICNoiseSize[2] != tdepth;
	}

	if (!this->LICNoiseTextureObject->GetHandle() || updateSize ||
		this->LICNoiseTextureObject->GetMTime() < this->Parent->LICNoiseGenerator->GetMTime())
	{
		int* winSize = ren->GetRenderWindow()->GetSize();
		int size[3];
		size[0] = this->Parent->LICNoiseSize[0];
		size[1] = this->Parent->LICNoiseSize[1];
		size[2] = this->Parent->LICNoiseSize[2];

		// Allocate buffer. After controlling for the maximum supported size sizeX/Y
		// might have changed, so an additional check is needed.
		int const twidth = this->LICNoiseTextureObject->GetWidth();
		int const theight = this->LICNoiseTextureObject->GetHeight();
		int const tdepth = this->LICNoiseTextureObject->GetDepth();
		bool sizeChanged = size[0] != twidth || size[1] != theight || size[2] != tdepth;

		// Generate random noise
		if (!this->Parent->LICNoiseGenerator)
		{
			// Use default settings
			vtkPerlinNoise* perlinNoise = vtkPerlinNoise::New();
			perlinNoise->SetPhase(0.5, 0.5, 0.5);
			perlinNoise->SetFrequency(size[0], size[1], size[2]);
			perlinNoise->SetAmplitude(0.5); /* [-n, n] */
			this->Parent->LICNoiseGenerator = perlinNoise;
		}
		if(!(this->LICNoiseTextureData))
			this->LICNoiseTextureData = (unsigned char*)LoadNoiseData("..\\..\\..\\VTKData\\noise\\noise_256_80", size);

		if (UseNoiseGradient && (this->LICNoiseTextureData) && !(this->LICNoiseGradientData))
			this->LICNoiseGradientData = (unsigned char*)loadGradients("..\\..\\..\\VTKData\\noise\\noise_256_80.grd", size);

		if (!(this->LICNoiseTextureData))
		{
			delete[] this->LICNoiseTextureData;
			this->LICNoiseTextureData = nullptr;
			this->LICNoiseTextureData = new unsigned char[size[0] * size[1] * size[2]];
			int const bufferSize = size[0] * size[1] * size[2];
			for (int x = 0; x < size[0]; x++)
				for (int y = 0; y < size[1]; y++)
					for (int z = 0; z < size[2]; z++)
					{
						int i = x + y * size[0] + z * size[0] * size[1];
						this->LICNoiseTextureData[i] = ((int)floor(0.6f*rand() / (RAND_MAX + 1.0f) + 0.5f)) * 255;
					}
		}
		// Prepare texture
		if (UseNoiseGradient)
		{
			unsigned char *packedData = new unsigned char[4 * size[0] * size[1] * size[2]];
			memset(packedData, 0, 4 * size[0] * size[1] * size[2] * sizeof(char));
			int adr, adrPacked;

			for (int z = 0; z<size[2]; ++z)
				for (int y = 0; y<size[1]; ++y)
					for (int x = 0; x<size[0]; ++x)
					{
						adrPacked = (z*size[1] + y)*size[0] + x;
						adr = (z*size[1] + y)*size[0] + x;
						assert(adr == adrPacked);
						packedData[4 * adrPacked + 0] = this->LICNoiseGradientData[3 * adr];
						packedData[4 * adrPacked + 1] = this->LICNoiseGradientData[3 * adr + 1];
						packedData[4 * adrPacked + 2] = this->LICNoiseGradientData[3 * adr + 2];
						packedData[4 * adrPacked + 3] = this->LICNoiseTextureData[adr];
					}
			this->LICNoiseTextureObject->Create3DFromRaw(size[0], size[1], size[2], 4, VTK_UNSIGNED_CHAR, packedData);
		}
		else
		{
			this->LICNoiseTextureObject->Create3DFromRaw(size[0], size[1], size[2], 1, VTK_UNSIGNED_CHAR, this->LICNoiseTextureData);
		}

		this->LICNoiseTextureObject->SetWrapS(vtkTextureObject::Repeat);
		this->LICNoiseTextureObject->SetWrapT(vtkTextureObject::Repeat);
		this->LICNoiseTextureObject->SetWrapR(vtkTextureObject::Repeat);
		this->LICNoiseTextureObject->SetMagnificationFilter(vtkTextureObject::Linear);
		this->LICNoiseTextureObject->SetMinificationFilter(vtkTextureObject::Linear);
		this->LICNoiseTextureObject->SetBorderColor(0.0f, 0.0f, 0.0f, 0.0f);
		this->LICNoiseTextureObject->Modified();
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::CreateKernelFilterTexture(vtkRenderer* ren)
{
	vtkOpenGLRenderWindow* glWindow = vtkOpenGLRenderWindow::SafeDownCast(
		ren->GetRenderWindow());

	if (!this->KernelFilterTextureObject)
	{
		this->KernelFilterTextureObject = vtkTextureObject::New();
	}
	this->KernelFilterTextureObject->SetContext(glWindow);

	// Allocate buffer. After controlling for the maximum supported size sizeX/Y
	// might have changed, so an additional check is needed.
	int sizeX = 256;
	if (!this->KernelFilterTextureObject->GetHandle())
	{
		double test[256];
		if (!this->KernelFilterData)
		{

			delete[] this->KernelFilterData;
			this->KernelFilterData = nullptr;
			this->KernelFilterData = new unsigned char[sizeX];
			

			for (int i = 0; i < sizeX; i++)
			{
				//test[i] = vtkMath::GaussianAmplitude(127, 2.0, i);
				if(i <= 127)
					this->KernelFilterData[i] = i * 2;
				else
					this->KernelFilterData[i] = (255 - i) * 2;
			}
		}

		this->KernelFilterTextureObject->Create1DFromRaw(sizeX, 1, VTK_UNSIGNED_CHAR,
			this->KernelFilterData);

		//this->KernelFilterTextureObject->SetWrapS(vtkTextureObject::ClampToBorder);
		this->KernelFilterTextureObject->SetWrapS(vtkTextureObject::ClampToEdge);
		this->KernelFilterTextureObject->SetMagnificationFilter(vtkTextureObject::Linear);
		this->KernelFilterTextureObject->SetMinificationFilter(vtkTextureObject::Linear);
		//this->KernelFilterTextureObject->SetBorderColor(0.0f, 0.0f, 0.0f, 0.0f);
		this->KernelFilterTextureObject->Modified();
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::CaptureDepthTexture(
	vtkRenderer* ren, vtkVolume* vtkNotUsed(vol))
{
	// Make sure our render window is the current OpenGL context
	ren->GetRenderWindow()->MakeCurrent();

	// Load required extensions for grabbing depth sampler buffer
	if (!this->LoadDepthTextureExtensionsSucceeded)
	{
		this->LoadRequireDepthTextureExtensions(ren->GetRenderWindow());
	}

	// If we can't load the necessary extensions, provide
	// feedback on why it failed.
	if (!this->LoadDepthTextureExtensionsSucceeded)
	{
		std::cerr << this->ExtensionsStringStream.str() << std::endl;
		return;
	}

	if (!this->DepthTextureObject)
	{
		this->DepthTextureObject = vtkTextureObject::New();
	}

	this->DepthTextureObject->SetContext(vtkOpenGLRenderWindow::SafeDownCast(
		ren->GetRenderWindow()));
	if (!this->DepthTextureObject->GetHandle())
	{
		// First set the parameters
		this->DepthTextureObject->SetWrapS(vtkTextureObject::ClampToEdge);
		this->DepthTextureObject->SetWrapT(vtkTextureObject::ClampToEdge);
		this->DepthTextureObject->SetMagnificationFilter(vtkTextureObject::Linear);
		this->DepthTextureObject->SetMinificationFilter(vtkTextureObject::Linear);
		this->DepthTextureObject->AllocateDepth(this->WindowSize[0],
			this->WindowSize[1],
			4);
	}

#if GL_ES_VERSION_3_0 != 1
	// currently broken on ES
	this->DepthTextureObject->CopyFromFrameBuffer(this->WindowLowerLeft[0],
		this->WindowLowerLeft[1],
		0, 0,
		this->WindowSize[0],
		this->WindowSize[1]);
#endif
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::SetLightingParameters(
	vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol)
{
	if (!ren || !prog || !vol)
	{
		return;
	}

	if (vol && !vol->GetProperty()->GetShade())
	{
		return;
	}

	prog->SetUniformi("in_twoSidedLighting", ren->GetTwoSidedLighting());

	// for lightkit case there are some parameters to set
	vtkCamera* cam = ren->GetActiveCamera();
	vtkTransform* viewTF = cam->GetModelViewTransformObject();

	// Bind some light settings
	int numberOfLights = 0;
	vtkLightCollection *lc = ren->GetLights();
	vtkLight *light;

	vtkCollectionSimpleIterator sit;
	float lightAmbientColor[6][3];
	float lightDiffuseColor[6][3];
	float lightSpecularColor[6][3];
	float lightDirection[6][3];
	for (lc->InitTraversal(sit);
		(light = lc->GetNextLight(sit)); )
	{
		float status = light->GetSwitch();
		if (status > 0.0)
		{
			double* aColor = light->GetAmbientColor();
			double* dColor = light->GetDiffuseColor();
			double* sColor = light->GetDiffuseColor();
			double intensity = light->GetIntensity();
			lightAmbientColor[numberOfLights][0] = aColor[0] * intensity;
			lightAmbientColor[numberOfLights][1] = aColor[1] * intensity;
			lightAmbientColor[numberOfLights][2] = aColor[2] * intensity;
			lightDiffuseColor[numberOfLights][0] = dColor[0] * intensity;
			lightDiffuseColor[numberOfLights][1] = dColor[1] * intensity;
			lightDiffuseColor[numberOfLights][2] = dColor[2] * intensity;
			lightSpecularColor[numberOfLights][0] = sColor[0] * intensity;
			lightSpecularColor[numberOfLights][1] = sColor[1] * intensity;
			lightSpecularColor[numberOfLights][2] = sColor[2] * intensity;
			// Get required info from light
			double* lfp = light->GetTransformedFocalPoint();
			double* lp = light->GetTransformedPosition();
			double lightDir[3];
			vtkMath::Subtract(lfp, lp, lightDir);
			vtkMath::Normalize(lightDir);
			double *tDir = viewTF->TransformNormal(lightDir);
			lightDirection[numberOfLights][0] = tDir[0];
			lightDirection[numberOfLights][1] = tDir[1];
			lightDirection[numberOfLights][2] = tDir[2];
			numberOfLights++;
		}
	}

	prog->SetUniform3fv("in_lightAmbientColor",
		numberOfLights, lightAmbientColor);
	prog->SetUniform3fv("in_lightDiffuseColor",
		numberOfLights, lightDiffuseColor);
	prog->SetUniform3fv("in_lightSpecularColor",
		numberOfLights, lightSpecularColor);
	prog->SetUniform3fv("in_lightDirection",
		numberOfLights, lightDirection);
	prog->SetUniformi("in_numberOfLights",
		numberOfLights);

	// we are done unless we have positional lights
	if (this->LightComplexity < 3)
	{
		return;
	}

	// if positional lights pass down more parameters
	float lightAttenuation[6][3];
	float lightPosition[6][3];
	float lightConeAngle[6];
	float lightExponent[6];
	int lightPositional[6];
	numberOfLights = 0;
	for (lc->InitTraversal(sit);
		(light = lc->GetNextLight(sit)); )
	{
		float status = light->GetSwitch();
		if (status > 0.0)
		{
			double* attn = light->GetAttenuationValues();
			lightAttenuation[numberOfLights][0] = attn[0];
			lightAttenuation[numberOfLights][1] = attn[1];
			lightAttenuation[numberOfLights][2] = attn[2];
			lightExponent[numberOfLights] = light->GetExponent();
			lightConeAngle[numberOfLights] = light->GetConeAngle();
			double* lp = light->GetTransformedPosition();
			double* tlp = viewTF->TransformPoint(lp);
			lightPosition[numberOfLights][0] = tlp[0];
			lightPosition[numberOfLights][1] = tlp[1];
			lightPosition[numberOfLights][2] = tlp[2];
			lightPositional[numberOfLights] = light->GetPositional();
			numberOfLights++;
		}
	}
	prog->SetUniform3fv("in_lightAttenuation", numberOfLights, lightAttenuation);
	prog->SetUniform1iv("in_lightPositional", numberOfLights, lightPositional);
	prog->SetUniform3fv("in_lightPosition", numberOfLights, lightPosition);
	prog->SetUniform1fv("in_lightExponent", numberOfLights, lightExponent);
	prog->SetUniform1fv("in_lightConeAngle", numberOfLights, lightConeAngle);
}

//-----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::ComputeCellToPointMatrix()
{
	this->CellToPointMatrix->Identity();
	this->AdjustedTexMin[0] = this->AdjustedTexMin[1] = this->AdjustedTexMin[2] = 0.0f;
	this->AdjustedTexMin[3] = 1.0f;
	this->AdjustedTexMax[0] = this->AdjustedTexMax[1] = this->AdjustedTexMax[2] = 1.0f;
	this->AdjustedTexMax[3] = 1.0f;

	if (!this->Parent->CellFlag) // point data
	{
		float delta[3];
		delta[0] = this->Extents[1] - this->Extents[0];
		delta[1] = this->Extents[3] - this->Extents[2];
		delta[2] = this->Extents[5] - this->Extents[4];

		float min[3];
		min[0] = 0.5f / delta[0];
		min[1] = 0.5f / delta[1];
		min[2] = 0.5f / delta[2];

		float range[3]; // max - min
		range[0] = (delta[0] - 0.5f) / delta[0] - min[0];
		range[1] = (delta[1] - 0.5f) / delta[1] - min[1];
		range[2] = (delta[2] - 0.5f) / delta[2] - min[2];

		this->CellToPointMatrix->SetElement(0, 0, range[0]); // Scale diag
		this->CellToPointMatrix->SetElement(1, 1, range[1]);
		this->CellToPointMatrix->SetElement(2, 2, range[2]);
		this->CellToPointMatrix->SetElement(0, 3, min[0]);   // t vector
		this->CellToPointMatrix->SetElement(1, 3, min[1]);
		this->CellToPointMatrix->SetElement(2, 3, min[2]);

		// Adjust limit coordinates for texture access.
		float const zeros[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; // GL tex min
		float const ones[4] = { 1.0f, 1.0f, 1.0f, 1.0f }; // GL tex max
		this->CellToPointMatrix->MultiplyPoint(zeros, this->AdjustedTexMin);
		this->CellToPointMatrix->MultiplyPoint(ones, this->AdjustedTexMax);
	}
}

//----------------------------------------------------------------------------
bool vtkLIC3DMapper::vtkInternal::IsCameraInside(
	vtkRenderer* ren, vtkVolume* vol)
{
	this->TempMatrix1->DeepCopy(vol->GetMatrix());
	this->TempMatrix1->Invert();

	vtkCamera* cam = ren->GetActiveCamera();
	double camWorldRange[2];
	double camWorldPos[4];
	double camFocalWorldPoint[4];
	double camWorldDirection[4];
	double camPos[4];
	double camPlaneNormal[4];

	cam->GetPosition(camWorldPos);
	camWorldPos[3] = 1.0;
	this->TempMatrix1->MultiplyPoint(camWorldPos, camPos);

	cam->GetFocalPoint(camFocalWorldPoint);
	camFocalWorldPoint[3] = 1.0;

	// The range (near/far) must also be transformed
	// into the local coordinate system.
	camWorldDirection[0] = camFocalWorldPoint[0] - camWorldPos[0];
	camWorldDirection[1] = camFocalWorldPoint[1] - camWorldPos[1];
	camWorldDirection[2] = camFocalWorldPoint[2] - camWorldPos[2];
	camWorldDirection[3] = 0.0;

	// Compute the normalized near plane normal
	this->TempMatrix1->MultiplyPoint(camWorldDirection, camPlaneNormal);

	vtkMath::Normalize(camWorldDirection);
	vtkMath::Normalize(camPlaneNormal);

	double camNearWorldPoint[4];
	double camNearPoint[4];

	cam->GetClippingRange(camWorldRange);
	camNearWorldPoint[0] = camWorldPos[0] + camWorldRange[0] * camWorldDirection[0];
	camNearWorldPoint[1] = camWorldPos[1] + camWorldRange[0] * camWorldDirection[1];
	camNearWorldPoint[2] = camWorldPos[2] + camWorldRange[0] * camWorldDirection[2];
	camNearWorldPoint[3] = 1.;

	this->TempMatrix1->MultiplyPoint(camNearWorldPoint, camNearPoint);

	int const result = vtkMath::PlaneIntersectsAABB(this->LoadedBounds,
		camPlaneNormal, camNearPoint);

	if (result == 0)
	{
		return true;
	}

	return false;
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::RenderVolumeGeometry(
	vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol)
{
	if (this->NeedToInitializeResources ||
		!this->BBoxPolyData ||
		this->Parent->VolumeTexture->UploadTime > this->BBoxPolyData->GetMTime() ||
		this->IsCameraInside(ren, vol) ||
		this->CameraWasInsideInLastUpdate)
	{
		vtkNew<vtkTessellatedBoxSource> boxSource;
		boxSource->SetBounds(this->LoadedBounds);
		boxSource->QuadsOn();
		boxSource->SetLevel(0);

		vtkNew<vtkDensifyPolyData> densityPolyData;

		if (this->IsCameraInside(ren, vol))
		{
			// Normals should be transformed using the transpose of inverse
			// InverseVolumeMat
			this->TempMatrix1->DeepCopy(vol->GetMatrix());
			this->TempMatrix1->Invert();

			vtkCamera* cam = ren->GetActiveCamera();
			double camWorldRange[2];
			double camWorldPos[4];
			double camFocalWorldPoint[4];
			double camWorldDirection[4];
			double camPos[4];
			double camPlaneNormal[4];

			cam->GetPosition(camWorldPos);
			camWorldPos[3] = 1.0;
			this->TempMatrix1->MultiplyPoint(camWorldPos, camPos);

			cam->GetFocalPoint(camFocalWorldPoint);
			camFocalWorldPoint[3] = 1.0;

			// The range (near/far) must also be transformed
			// into the local coordinate system.
			camWorldDirection[0] = camFocalWorldPoint[0] - camWorldPos[0];
			camWorldDirection[1] = camFocalWorldPoint[1] - camWorldPos[1];
			camWorldDirection[2] = camFocalWorldPoint[2] - camWorldPos[2];
			camWorldDirection[3] = 0.0;

			// Compute the normalized near plane normal
			this->TempMatrix1->MultiplyPoint(camWorldDirection, camPlaneNormal);

			vtkMath::Normalize(camWorldDirection);
			vtkMath::Normalize(camPlaneNormal);

			double camNearWorldPoint[4];
			double camFarWorldPoint[4];
			double camNearPoint[4];
			double camFarPoint[4];

			cam->GetClippingRange(camWorldRange);
			camNearWorldPoint[0] = camWorldPos[0] + camWorldRange[0] * camWorldDirection[0];
			camNearWorldPoint[1] = camWorldPos[1] + camWorldRange[0] * camWorldDirection[1];
			camNearWorldPoint[2] = camWorldPos[2] + camWorldRange[0] * camWorldDirection[2];
			camNearWorldPoint[3] = 1.;

			camFarWorldPoint[0] = camWorldPos[0] + camWorldRange[1] * camWorldDirection[0];
			camFarWorldPoint[1] = camWorldPos[1] + camWorldRange[1] * camWorldDirection[1];
			camFarWorldPoint[2] = camWorldPos[2] + camWorldRange[1] * camWorldDirection[2];
			camFarWorldPoint[3] = 1.;

			this->TempMatrix1->MultiplyPoint(camNearWorldPoint, camNearPoint);
			this->TempMatrix1->MultiplyPoint(camFarWorldPoint, camFarPoint);

			vtkNew<vtkPlane> nearPlane;

			// We add an offset to the near plane to avoid hardware clipping of the
			// near plane due to floating-point precision.
			// camPlaneNormal is a unit vector, if the offset is larger than the
			// distance between near and far point, it will not work. Hence, we choose
			// a fraction of the near-far distance. However, care should be taken
			// to avoid hardware clipping in volumes with very small spacing where the
			// distance between near and far plane is also very small. In that case,
			// a minimum offset is chosen. This is chosen based on the typical
			// epsilon values on x86 systems.
			double offset = sqrt(vtkMath::Distance2BetweenPoints(
				camNearPoint, camFarPoint)) / 1000.0;
			// Minimum offset to avoid floating point precision issues for volumes
			// with very small spacing
			double minOffset = static_cast<double>(
				std::numeric_limits<float>::epsilon()) * 1000.0;
			offset = offset < minOffset ? minOffset : offset;

			camNearPoint[0] += camPlaneNormal[0] * offset;
			camNearPoint[1] += camPlaneNormal[1] * offset;
			camNearPoint[2] += camPlaneNormal[2] * offset;

			nearPlane->SetOrigin(camNearPoint);
			nearPlane->SetNormal(camPlaneNormal);

			vtkNew<vtkPlaneCollection> planes;
			planes->RemoveAllItems();
			planes->AddItem(nearPlane);

			vtkNew<vtkClipConvexPolyData> clip;
			clip->SetInputConnection(boxSource->GetOutputPort());
			clip->SetPlanes(planes);

			densityPolyData->SetInputConnection(clip->GetOutputPort());

			this->CameraWasInsideInLastUpdate = true;
		}
		else
		{
			densityPolyData->SetInputConnection(boxSource->GetOutputPort());
			this->CameraWasInsideInLastUpdate = false;
		}

		densityPolyData->SetNumberOfSubdivisions(2);
		densityPolyData->Update();

		this->BBoxPolyData = vtkSmartPointer<vtkPolyData>::New();
		this->BBoxPolyData->ShallowCopy(densityPolyData->GetOutput());
		vtkPoints* points = this->BBoxPolyData->GetPoints();
		vtkCellArray* cells = this->BBoxPolyData->GetPolys();

		vtkNew<vtkUnsignedIntArray> polys;
		polys->SetNumberOfComponents(3);
		vtkIdType npts;
		vtkIdType *pts;

		// See if the volume transform is orientation-preserving
		// and orient polygons accordingly
		vtkMatrix4x4* volMat = vol->GetMatrix();
		double det = vtkMath::Determinant3x3(
			volMat->GetElement(0, 0), volMat->GetElement(0, 1), volMat->GetElement(0, 2),
			volMat->GetElement(1, 0), volMat->GetElement(1, 1), volMat->GetElement(1, 2),
			volMat->GetElement(2, 0), volMat->GetElement(2, 1), volMat->GetElement(2, 2));
		bool preservesOrientation = det > 0.0;

		const vtkIdType indexMap[3] = {
			preservesOrientation ? 0 : 2,
			1,
			preservesOrientation ? 2 : 0
		};

		while (cells->GetNextCell(npts, pts))
		{
			polys->InsertNextTuple3(pts[indexMap[0]], pts[indexMap[1]], pts[indexMap[2]]);
		}

		// Dispose any previously created buffers
		this->DeleteBufferObjects();

		// Now create new ones
		this->CreateBufferObjects();

		// TODO: should really use the built in VAO class
		// which handles these apple issues internally
#ifdef __APPLE__
		if (vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
#endif
		{
			glBindVertexArray(this->CubeVAOId);
	}

		// Pass cube vertices to buffer object memory
		glBindBuffer(GL_ARRAY_BUFFER, this->CubeVBOId);
		glBufferData(GL_ARRAY_BUFFER, points->GetData()->GetDataSize() *
			points->GetData()->GetDataTypeSize(),
			points->GetData()->GetVoidPointer(0), GL_STATIC_DRAW);

		prog->EnableAttributeArray("in_vertexPos");
		prog->UseAttributeArray("in_vertexPos", 0, 0, VTK_FLOAT,
			3, vtkShaderProgram::NoNormalize);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->CubeIndicesId);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, polys->GetDataSize() *
			polys->GetDataTypeSize(), polys->GetVoidPointer(0),
			GL_STATIC_DRAW);
}
	else
	{
#ifdef __APPLE__
		if (!vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
		{
			glBindBuffer(GL_ARRAY_BUFFER, this->CubeVBOId);
			prog->EnableAttributeArray("in_vertexPos");
			prog->UseAttributeArray("in_vertexPos", 0, 0, VTK_FLOAT,
				3, vtkShaderProgram::NoNormalize);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->CubeIndicesId);
		}
		else
#endif
		{
			glBindVertexArray(this->CubeVAOId);
		}
	}

	glDrawElements(GL_TRIANGLES,
		this->BBoxPolyData->GetNumberOfCells() * 3,
		GL_UNSIGNED_INT, nullptr);

	vtkOpenGLStaticCheckErrorMacro("Error after glDrawElements in"
		" RenderVolumeGeometry!");
#ifdef __APPLE__
	if (!vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
	{
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
	else
#endif
	{
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::SetCroppingRegions(
	vtkRenderer* vtkNotUsed(ren), vtkShaderProgram* prog,
	vtkVolume* vtkNotUsed(vol))
{
	if (this->Parent->GetCropping())
	{
		int cropFlags = this->Parent->GetCroppingRegionFlags();
		double croppingRegionPlanes[6];
		this->Parent->GetCroppingRegionPlanes(croppingRegionPlanes);

		// Clamp it
		croppingRegionPlanes[0] = croppingRegionPlanes[0] < this->LoadedBounds[0] ?
			this->LoadedBounds[0] : croppingRegionPlanes[0];
		croppingRegionPlanes[0] = croppingRegionPlanes[0] > this->LoadedBounds[1] ?
			this->LoadedBounds[1] : croppingRegionPlanes[0];
		croppingRegionPlanes[1] = croppingRegionPlanes[1] < this->LoadedBounds[0] ?
			this->LoadedBounds[0] : croppingRegionPlanes[1];
		croppingRegionPlanes[1] = croppingRegionPlanes[1] > this->LoadedBounds[1] ?
			this->LoadedBounds[1] : croppingRegionPlanes[1];

		croppingRegionPlanes[2] = croppingRegionPlanes[2] < this->LoadedBounds[2] ?
			this->LoadedBounds[2] : croppingRegionPlanes[2];
		croppingRegionPlanes[2] = croppingRegionPlanes[2] > this->LoadedBounds[3] ?
			this->LoadedBounds[3] : croppingRegionPlanes[2];
		croppingRegionPlanes[3] = croppingRegionPlanes[3] < this->LoadedBounds[2] ?
			this->LoadedBounds[2] : croppingRegionPlanes[3];
		croppingRegionPlanes[3] = croppingRegionPlanes[3] > this->LoadedBounds[3] ?
			this->LoadedBounds[3] : croppingRegionPlanes[3];

		croppingRegionPlanes[4] = croppingRegionPlanes[4] < this->LoadedBounds[4] ?
			this->LoadedBounds[4] : croppingRegionPlanes[4];
		croppingRegionPlanes[4] = croppingRegionPlanes[4] > this->LoadedBounds[5] ?
			this->LoadedBounds[5] : croppingRegionPlanes[4];
		croppingRegionPlanes[5] = croppingRegionPlanes[5] < this->LoadedBounds[4] ?
			this->LoadedBounds[4] : croppingRegionPlanes[5];
		croppingRegionPlanes[5] = croppingRegionPlanes[5] > this->LoadedBounds[5] ?
			this->LoadedBounds[5] : croppingRegionPlanes[5];

		float cropPlanes[6] = { static_cast<float>(croppingRegionPlanes[0]),
			static_cast<float>(croppingRegionPlanes[1]),
			static_cast<float>(croppingRegionPlanes[2]),
			static_cast<float>(croppingRegionPlanes[3]),
			static_cast<float>(croppingRegionPlanes[4]),
			static_cast<float>(croppingRegionPlanes[5]) };

		prog->SetUniform1fv("in_croppingPlanes", 6, cropPlanes);
		const int numberOfRegions = 32;
		int cropFlagsArray[numberOfRegions];
		cropFlagsArray[0] = 0;
		int i = 1;
		while (cropFlags && i < 32)
		{
			cropFlagsArray[i] = cropFlags & 1;
			cropFlags = cropFlags >> 1;
			++i;
		}
		for (; i < 32; ++i)
		{
			cropFlagsArray[i] = 0;
		}

		prog->SetUniform1iv("in_croppingFlags", numberOfRegions, cropFlagsArray);
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::SetClippingPlanes(
	vtkRenderer* vtkNotUsed(ren), vtkShaderProgram* prog,
	vtkVolume* vtkNotUsed(vol))
{
	if (this->Parent->GetClippingPlanes())
	{
		std::vector<float> clippingPlanes;
		// Currently we don't have any clipping plane
		clippingPlanes.push_back(0);

		this->Parent->ClippingPlanes->InitTraversal();
		vtkPlane* plane;
		while ((plane = this->Parent->ClippingPlanes->GetNextItem()))
		{
			// Planes are in world coordinates
			double planeOrigin[3], planeNormal[3];
			plane->GetOrigin(planeOrigin);
			plane->GetNormal(planeNormal);

			clippingPlanes.push_back(planeOrigin[0]);
			clippingPlanes.push_back(planeOrigin[1]);
			clippingPlanes.push_back(planeOrigin[2]);
			clippingPlanes.push_back(planeNormal[0]);
			clippingPlanes.push_back(planeNormal[1]);
			clippingPlanes.push_back(planeNormal[2]);
		}

		clippingPlanes[0] = clippingPlanes.size() > 1 ?
			static_cast<int>(clippingPlanes.size() - 1) : 0;

		prog->SetUniform1fv("in_clippingPlanes",
			static_cast<int>(clippingPlanes.size()),
			&clippingPlanes[0]);
	}
}

// -----------------------------------------------------------------------------
void
vtkLIC3DMapper::vtkInternal::CheckPropertyKeys(vtkVolume* vol)
{
	// Check the property keys to see if we should modify the blend/etc state:
	// Otherwise this breaks volume/translucent geo depth peeling.
	vtkInformation *volumeKeys = vol->GetPropertyKeys();
	this->PreserveGLState = false;
	if (volumeKeys && volumeKeys->Has(vtkOpenGLActor::GLDepthMaskOverride()))
	{
		int override = volumeKeys->Get(vtkOpenGLActor::GLDepthMaskOverride());
		if (override != 0 && override != 1)
		{
			this->PreserveGLState = true;
		}
	}

	// Some render passes (e.g. DualDepthPeeling) adjust the viewport for
	// intermediate passes so it is necessary to preserve it. This is a
	// temporary fix for vtkDualDepthPeelingPass to work when various viewports
	// are defined.  The correct way of fixing this would be to avoid setting the
	// viewport within the mapper.  It is enough for now to check for the
	// RenderPasses() vtkInfo given that vtkDualDepthPeelingPass is the only pass
	// currently supported by this mapper, the viewport will have to be adjusted
	// externally before adding support for other passes.
	vtkInformation *info = vol->GetPropertyKeys();
	this->PreserveViewport = info && info->Has(
		vtkOpenGLRenderPass::RenderPasses());
}

// -----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::CheckPickingState(vtkRenderer* ren)
{
	vtkHardwareSelector* selector = ren->GetSelector();
	bool selectorPicking = selector != nullptr;
	if (selector)
	{
		// this mapper currently only supports cell picking
		selectorPicking &= selector->GetFieldAssociation() == vtkDataObject::FIELD_ASSOCIATION_CELLS;
	}

	this->IsPicking = selectorPicking || ren->GetRenderWindow()->GetIsPicking();
	if (this->IsPicking)
	{
		// rebuild the shader on every pass
		this->SelectionStateTime.Modified();
		this->CurrentSelectionPass = selector ? selector->GetCurrentPass() : vtkHardwareSelector::ACTOR_PASS;
	}
	else if (this->CurrentSelectionPass != vtkHardwareSelector::MIN_KNOWN_PASS - 1)
	{
		// return to the regular rendering state
		this->SelectionStateTime.Modified();
		this->CurrentSelectionPass = vtkHardwareSelector::MIN_KNOWN_PASS - 1;
	}
}

// -----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::BeginPicking(vtkRenderer* ren)
{
	vtkHardwareSelector* selector = ren->GetSelector();
	if (selector && this->IsPicking)
	{
		selector->BeginRenderProp();

		if (this->CurrentSelectionPass >= vtkHardwareSelector::ID_LOW24)
		{
			selector->RenderAttributeId(0);
		}
	}
}

//------------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::SetPickingId
(vtkRenderer* ren)
{
	float propIdColor[3] = { 0.0, 0.0, 0.0 };
	vtkHardwareSelector* selector = ren->GetSelector();

	if (selector && this->IsPicking)
	{
		// query the selector for the appropriate id
		selector->GetPropColorValue(propIdColor);
	}
	else // RenderWindow is picking
	{
		unsigned int const idx = ren->GetCurrentPickId();
		vtkHardwareSelector::Convert(idx, propIdColor);
	}

	this->ShaderProgram->SetUniform3f("in_propId", propIdColor);
}

// ---------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::EndPicking(vtkRenderer* ren)
{
	vtkHardwareSelector* selector = ren->GetSelector();
	if (selector && this->IsPicking)
	{
		if (this->CurrentSelectionPass >= vtkHardwareSelector::ID_LOW24)
		{
			// tell the selector the maximum number of cells that the mapper could render
			unsigned int const numVoxels = (this->Extents[1] - this->Extents[0]) *
				(this->Extents[3] - this->Extents[2]) * (this->Extents[5] - this->Extents[4]);
			selector->RenderAttributeId(numVoxels);
		}
		selector->EndRenderProp();
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::UpdateSamplingDistance(
	vtkImageData* input, vtkRenderer* vtkNotUsed(ren), vtkVolume* vol)
{
	if (!this->Parent->AutoAdjustSampleDistances)
	{
		if (this->Parent->LockSampleDistanceToInputSpacing)
		{
			float const d = static_cast<float>(this->Parent->SpacingAdjustedSampleDistance(
				this->CellSpacing, this->Extents));
			float const sample = this->Parent->SampleDistance;

			// ActualSampleDistance will grow proportionally to numVoxels^(1/3) (see
			// vtkVolumeMapper.cxx). Until it reaches 1/2 average voxel size when number of
			// voxels is 1E6.
			this->ActualSampleDistance = (sample / d < 0.999f || sample / d > 1.001f) ?
				d : this->Parent->SampleDistance;

			return;
		}

		this->ActualSampleDistance = this->Parent->SampleDistance;
	}
	else
	{
		input->GetSpacing(this->CellSpacing);
		vtkMatrix4x4* worldToDataset = vol->GetMatrix();
		double minWorldSpacing = VTK_DOUBLE_MAX;
		int i = 0;
		while (i < 3)
		{
			double tmp = worldToDataset->GetElement(0, i);
			double tmp2 = tmp * tmp;
			tmp = worldToDataset->GetElement(1, i);
			tmp2 += tmp * tmp;
			tmp = worldToDataset->GetElement(2, i);
			tmp2 += tmp * tmp;

			// We use fabs() in case the spacing is negative.
			double worldSpacing = fabs(this->CellSpacing[i] * sqrt(tmp2));
			if (worldSpacing < minWorldSpacing)
			{
				minWorldSpacing = worldSpacing;
			}
			++i;
		}

		// minWorldSpacing is the optimal sample distance in world space.
		// To go faster (reduceFactor<1.0), we multiply this distance
		// by 1/reduceFactor.
		this->ActualSampleDistance = static_cast<float>(minWorldSpacing);

		if (this->Parent->ReductionFactor < 1.0 &&
			this->Parent->ReductionFactor != 0.0)
		{
			this->ActualSampleDistance /=
				static_cast<GLfloat>(this->Parent->ReductionFactor);
		}
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::
LoadRequireDepthTextureExtensions(vtkRenderWindow* vtkNotUsed(renWin))
{
	// Reset the message stream for extensions
	if (vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
	{
		this->LoadDepthTextureExtensionsSucceeded = true;
		return;
	}

	this->ExtensionsStringStream.str("");
	this->ExtensionsStringStream.clear();

#if GL_ES_VERSION_3_0 != 1
	// Check for float texture support. This extension became core
	// in 3.0
	if (!glewIsSupported("GL_ARB_texture_float"))
	{
		this->ExtensionsStringStream << "Required extension "
			<< " GL_ARB_texture_float is not supported";
		return;
	}
#endif

	// NOTE: Support for depth sampler texture made into the core since version
	// 1.4 and therefore we are no longer checking for it.
	this->LoadDepthTextureExtensionsSucceeded = true;
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::CreateBufferObjects()
{
#ifdef __APPLE__
	if (vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
#endif
	{
		glGenVertexArrays(1, &this->CubeVAOId);
}
	glGenBuffers(1, &this->CubeVBOId);
	glGenBuffers(1, &this->CubeIndicesId);
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::DeleteBufferObjects()
{
	if (this->CubeVBOId)
	{
		glBindBuffer(GL_ARRAY_BUFFER, this->CubeVBOId);
		glDeleteBuffers(1, &this->CubeVBOId);
		this->CubeVBOId = 0;
	}

	if (this->CubeIndicesId)
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->CubeIndicesId);
		glDeleteBuffers(1, &this->CubeIndicesId);
		this->CubeIndicesId = 0;
	}

	if (this->CubeVAOId)
	{
#ifdef __APPLE__
		if (vtkOpenGLRenderWindow::GetContextSupportsOpenGL32())
#endif
		{
			glDeleteVertexArrays(1, &this->CubeVAOId);
	}
		this->CubeVAOId = 0;
}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::
ConvertTextureToImageData(vtkTextureObject* texture, vtkImageData* output)
{
	if (!texture)
	{
		return;
	}
	unsigned int tw = texture->GetWidth();
	unsigned int th = texture->GetHeight();
	unsigned int tnc = texture->GetComponents();
	int tt = texture->GetVTKDataType();

	vtkPixelExtent texExt(0U, tw - 1U, 0U, th - 1U);

	int dataExt[6] = { 0,0, 0,0, 0,0 };
	texExt.GetData(dataExt);

	double dataOrigin[6] = { 0, 0, 0, 0, 0, 0 };

	vtkImageData *id = vtkImageData::New();
	id->SetOrigin(dataOrigin);
	id->SetDimensions(tw, th, 1);
	id->SetExtent(dataExt);
	id->AllocateScalars(tt, tnc);

	vtkPixelBufferObject *pbo = texture->Download();

	vtkPixelTransfer::Blit(texExt,
		texExt,
		texExt,
		texExt,
		tnc,
		tt,
		pbo->MapPackedBuffer(),
		tnc,
		tt,
		id->GetScalarPointer(0, 0, 0));

	pbo->UnmapPackedBuffer();
	pbo->Delete();

	if (!output)
	{
		output = vtkImageData::New();
	}
	output->DeepCopy(id);
	id->Delete();
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::BeginImageSample(
	vtkRenderer* ren, vtkVolume* vol)
{
	const auto numBuffers = this->GetNumImageSampleDrawBuffers(vol);
	if (numBuffers != this->NumImageSampleDrawBuffers)
	{
		if (numBuffers > this->NumImageSampleDrawBuffers)
		{
			this->ReleaseImageSampleGraphicsResources(ren->GetRenderWindow());
		}

		this->NumImageSampleDrawBuffers = numBuffers;
		this->RebuildImageSampleProg = true;
	}

	float const xySampleDist = this->Parent->ImageSampleDistance;
	if (xySampleDist != 1.f && this->InitializeImageSampleFBO(ren))
	{
		this->ImageSampleFBO->SaveCurrentBindingsAndBuffers(GL_DRAW_FRAMEBUFFER);
		this->ImageSampleFBO->DeactivateDrawBuffers();
		this->ImageSampleFBO->Bind(GL_DRAW_FRAMEBUFFER);
		this->ImageSampleFBO->ActivateDrawBuffers(static_cast<unsigned int>(
			this->NumImageSampleDrawBuffers));

		glClearColor(0.0, 0.0, 0.0, 0.0);
		glClear(GL_COLOR_BUFFER_BIT);
	}
}

//----------------------------------------------------------------------------
bool vtkLIC3DMapper::vtkInternal::InitializeImageSampleFBO(
	vtkRenderer* ren)
{
	// Set the FBO viewport size. These are used in the shader to normalize the
	// fragment coordinate, the normalized coordinate is used to fetch the depth
	// buffer.
	this->WindowSize[0] /= this->Parent->ImageSampleDistance;
	this->WindowSize[1] /= this->Parent->ImageSampleDistance;
	this->WindowLowerLeft[0] = 0;
	this->WindowLowerLeft[1] = 0;

	// Set FBO viewport
	glViewport(this->WindowLowerLeft[0], this->WindowLowerLeft[1],
		this->WindowSize[0], this->WindowSize[1]);

	if (!this->ImageSampleFBO)
	{
		vtkOpenGLRenderWindow* win = vtkOpenGLRenderWindow::SafeDownCast(
			ren->GetRenderWindow());

		this->ImageSampleTexture.reserve(this->NumImageSampleDrawBuffers);
		this->ImageSampleTexNames.reserve(this->NumImageSampleDrawBuffers);
		for (size_t i = 0; i < this->NumImageSampleDrawBuffers; i++)
		{
			auto tex = vtkSmartPointer<vtkTextureObject>::New();
			tex->SetContext(win);
			tex->Create2D(this->WindowSize[0], this->WindowSize[1],
				4, VTK_UNSIGNED_CHAR, false);
			tex->Activate();
			tex->SetMinificationFilter(vtkTextureObject::Linear);
			tex->SetMagnificationFilter(vtkTextureObject::Linear);
			tex->SetWrapS(vtkTextureObject::ClampToEdge);
			tex->SetWrapT(vtkTextureObject::ClampToEdge);
			this->ImageSampleTexture.push_back(tex);

			std::stringstream ss; ss << i;
			const std::string name = "renderedTex_" + ss.str();
			this->ImageSampleTexNames.push_back(name);
		}

		this->ImageSampleFBO = vtkOpenGLFramebufferObject::New();
		this->ImageSampleFBO->SetContext(win);
		this->ImageSampleFBO->SaveCurrentBindingsAndBuffers(GL_FRAMEBUFFER);
		this->ImageSampleFBO->Bind(GL_FRAMEBUFFER);
		this->ImageSampleFBO->InitializeViewport(this->WindowSize[0],
			this->WindowSize[1]);

		auto num = static_cast<unsigned int>(this->NumImageSampleDrawBuffers);
		for (unsigned int i = 0; i < num; i++)
		{
			this->ImageSampleFBO->AddColorAttachment(GL_FRAMEBUFFER, i,
				this->ImageSampleTexture[i]);
		}

		// Verify completeness
		const int complete = this->ImageSampleFBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);
		for (auto& tex : this->ImageSampleTexture)
		{
			tex->Deactivate();
		}
		this->ImageSampleFBO->RestorePreviousBindingsAndBuffers(GL_FRAMEBUFFER);

		if (!complete)
		{
			vtkGenericWarningMacro(<< "Failed to attach ImageSampleFBO!");
			this->ReleaseImageSampleGraphicsResources(win);
			return false;
		}

		this->RebuildImageSampleProg = true;
		return true;
	}

	// Resize if necessary
	int lastSize[2];
	this->ImageSampleFBO->GetLastSize(lastSize);
	if (lastSize[0] != this->WindowSize[0] || lastSize[1] != this->WindowSize[1])
	{
		this->ImageSampleFBO->Resize(this->WindowSize[0], this->WindowSize[1]);
	}

	return true;
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::EndImageSample(
	vtkRenderer* ren)
{
	if (this->Parent->ImageSampleDistance != 1.f)
	{
		this->ImageSampleFBO->DeactivateDrawBuffers();
		this->ImageSampleFBO->RestorePreviousBindingsAndBuffers(GL_DRAW_FRAMEBUFFER);
		if (this->RenderPassAttached)
		{
			this->ImageSampleFBO->ActivateDrawBuffers(static_cast<unsigned int>(
				this->NumImageSampleDrawBuffers));
		}

		// Render the contents of ImageSampleFBO as a quad to intermix with the
		// rest of the scene.
		typedef vtkOpenGLRenderUtilities GLUtil;
		vtkOpenGLRenderWindow* win = static_cast<vtkOpenGLRenderWindow*>(
			ren->GetRenderWindow());

		if (this->RebuildImageSampleProg)
		{
			std::string frag = GLUtil::GetFullScreenQuadFragmentShaderTemplate();

			vtkShaderProgram::Substitute(frag, "//VTK::FSQ::Decl",
				vtkvolume::ImageSampleDeclarationFrag(this->ImageSampleTexNames,
					this->NumImageSampleDrawBuffers));
			vtkShaderProgram::Substitute(frag, "//VTK::FSQ::Impl",
				vtkvolume::ImageSampleImplementationFrag(this->ImageSampleTexNames,
					this->NumImageSampleDrawBuffers));

			this->ImageSampleProg = win->GetShaderCache()->ReadyShaderProgram(
				GLUtil::GetFullScreenQuadVertexShader().c_str(), frag.c_str(),
				GLUtil::GetFullScreenQuadGeometryShader().c_str());
		}
		else
		{
			win->GetShaderCache()->ReadyShaderProgram(this->ImageSampleProg);
		}

		if (!this->ImageSampleProg)
		{
			vtkGenericWarningMacro(<< "Failed to initialize ImageSampleProgram!");
			return;
		}

		if (!this->ImageSampleVAO)
		{
			this->ImageSampleVBO = vtkOpenGLBufferObject::New();
			this->ImageSampleVAO = vtkOpenGLVertexArrayObject::New();
			GLUtil::PrepFullScreenVAO(this->ImageSampleVBO, this->ImageSampleVAO,
				this->ImageSampleProg);
		}

		// Adjust the GL viewport to VTK's defined viewport
		ren->GetTiledSizeAndOrigin(this->WindowSize, this->WindowSize + 1,
			this->WindowLowerLeft, this->WindowLowerLeft + 1);
		glViewport(this->WindowLowerLeft[0], this->WindowLowerLeft[1],
			this->WindowSize[0], this->WindowSize[1]);

		// Bind objects and draw
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_DEPTH_TEST);

		for (size_t i = 0; i < this->NumImageSampleDrawBuffers; i++)
		{
			this->ImageSampleTexture[i]->Activate();
			this->ImageSampleProg->SetUniformi(this->ImageSampleTexNames[i].c_str(),
				this->ImageSampleTexture[i]->GetTextureUnit());
		}

		this->ImageSampleVAO->Bind();
		GLUtil::DrawFullScreenQuad();
		this->ImageSampleVAO->Release();
		vtkOpenGLStaticCheckErrorMacro("Error after DrawFullScreenQuad()!");

		for (auto& tex : this->ImageSampleTexture)
		{
			tex->Deactivate();
		}
	}
}

//------------------------------------------------------------------------------
size_t vtkLIC3DMapper::vtkInternal::GetNumImageSampleDrawBuffers(
	vtkVolume* vol)
{
	if (this->RenderPassAttached)
	{
		vtkInformation* info = vol->GetPropertyKeys();
		const int num = info->Length(vtkOpenGLRenderPass::RenderPasses());
		vtkObjectBase *rpBase = info->Get(vtkOpenGLRenderPass::RenderPasses(), num - 1);
		vtkOpenGLRenderPass *rp = static_cast<vtkOpenGLRenderPass*>(rpBase);
		return static_cast<size_t>(rp->GetActiveDrawBuffers());
	}

	return 1;
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::SetupRenderToTexture(
	vtkRenderer* ren)
{
	if (this->Parent->RenderToImage && this->Parent->CurrentPass == RenderPass)
	{
		if (this->Parent->ImageSampleDistance != 1.f)
		{
			this->WindowSize[0] /= this->Parent->ImageSampleDistance;
			this->WindowSize[1] /= this->Parent->ImageSampleDistance;
		}

		if ((this->LastRenderToImageWindowSize[0] != this->WindowSize[0]) ||
			(this->LastRenderToImageWindowSize[1] != this->WindowSize[1]))
		{
			this->LastRenderToImageWindowSize[0] = this->WindowSize[0];
			this->LastRenderToImageWindowSize[1] = this->WindowSize[1];
			this->ReleaseRenderToTextureGraphicsResources(ren->GetRenderWindow());
		}

		if (!this->FBO)
		{
			this->FBO = vtkOpenGLFramebufferObject::New();
		}

		this->FBO->SetContext(vtkOpenGLRenderWindow::SafeDownCast(
			ren->GetRenderWindow()));

		this->FBO->SaveCurrentBindingsAndBuffers();
		this->FBO->Bind(GL_FRAMEBUFFER);
		this->FBO->InitializeViewport(this->WindowSize[0], this->WindowSize[1]);

		int depthImageScalarType = this->Parent->GetDepthImageScalarType();
		bool initDepthTexture = true;
		// Re-instantiate the depth texture object if the scalar type requested has
		// changed from the last frame
		if (this->RTTDepthTextureObject &&
			this->RTTDepthTextureType == depthImageScalarType)
		{
			initDepthTexture = false;
		}

		if (initDepthTexture)
		{
			if (this->RTTDepthTextureObject)
			{
				this->RTTDepthTextureObject->Delete();
				this->RTTDepthTextureObject = nullptr;
			}
			this->RTTDepthTextureObject = vtkTextureObject::New();
			this->RTTDepthTextureObject->SetContext(
				vtkOpenGLRenderWindow::SafeDownCast(
					ren->GetRenderWindow()));
			this->RTTDepthTextureObject->Create2D(this->WindowSize[0],
				this->WindowSize[1], 1,
				depthImageScalarType, false);
			this->RTTDepthTextureObject->Activate();
			this->RTTDepthTextureObject->SetMinificationFilter(
				vtkTextureObject::Nearest);
			this->RTTDepthTextureObject->SetMagnificationFilter(
				vtkTextureObject::Nearest);
			this->RTTDepthTextureObject->SetAutoParameters(0);

			// Cache the value of the scalar type
			this->RTTDepthTextureType = depthImageScalarType;
		}

		if (!this->RTTColorTextureObject)
		{
			this->RTTColorTextureObject = vtkTextureObject::New();

			this->RTTColorTextureObject->SetContext(
				vtkOpenGLRenderWindow::SafeDownCast(
					ren->GetRenderWindow()));
			this->RTTColorTextureObject->Create2D(this->WindowSize[0],
				this->WindowSize[1], 4,
				VTK_UNSIGNED_CHAR, false);
			this->RTTColorTextureObject->Activate();
			this->RTTColorTextureObject->SetMinificationFilter(
				vtkTextureObject::Nearest);
			this->RTTColorTextureObject->SetMagnificationFilter(
				vtkTextureObject::Nearest);
			this->RTTColorTextureObject->SetAutoParameters(0);
		}

		if (!this->RTTDepthBufferTextureObject)
		{
			this->RTTDepthBufferTextureObject = vtkTextureObject::New();
			this->RTTDepthBufferTextureObject->SetContext(
				vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));
			this->RTTDepthBufferTextureObject->AllocateDepth(
				this->WindowSize[0], this->WindowSize[1], vtkTextureObject::Float32);
			this->RTTDepthBufferTextureObject->Activate();
			this->RTTDepthBufferTextureObject->SetMinificationFilter(
				vtkTextureObject::Nearest);
			this->RTTDepthBufferTextureObject->SetMagnificationFilter(
				vtkTextureObject::Nearest);
			this->RTTDepthBufferTextureObject->SetAutoParameters(0);
		}

		this->FBO->Bind(GL_FRAMEBUFFER);
		this->FBO->AddDepthAttachment(
			GL_FRAMEBUFFER,
			this->RTTDepthBufferTextureObject);
		this->FBO->AddColorAttachment(
			GL_FRAMEBUFFER, 0U,
			this->RTTColorTextureObject);
		this->FBO->AddColorAttachment(
			GL_FRAMEBUFFER, 1U,
			this->RTTDepthTextureObject);
		this->FBO->ActivateDrawBuffers(2);

		this->FBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);

		glClearColor(1.0, 1.0, 1.0, 0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::ExitRenderToTexture(
	vtkRenderer* vtkNotUsed(ren))
{
	if (this->Parent->RenderToImage && this->Parent->CurrentPass == RenderPass)
	{
		this->FBO->RemoveTexDepthAttachment(GL_FRAMEBUFFER);
		this->FBO->RemoveTexColorAttachment(GL_FRAMEBUFFER, 0U);
		this->FBO->RemoveTexColorAttachment(GL_FRAMEBUFFER, 1U);
		this->FBO->DeactivateDrawBuffers();
		this->FBO->RestorePreviousBindingsAndBuffers();

		this->RTTDepthBufferTextureObject->Deactivate();
		this->RTTColorTextureObject->Deactivate();
		this->RTTDepthTextureObject->Deactivate();
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::SetupDepthPass(
	vtkRenderer* ren)
{
	if (this->Parent->ImageSampleDistance != 1.f)
	{
		this->WindowSize[0] /= this->Parent->ImageSampleDistance;
		this->WindowSize[1] /= this->Parent->ImageSampleDistance;
	}

	if ((this->LastDepthPassWindowSize[0] != this->WindowSize[0]) ||
		(this->LastDepthPassWindowSize[1] != this->WindowSize[1]))
	{
		this->LastDepthPassWindowSize[0] = this->WindowSize[0];
		this->LastDepthPassWindowSize[1] = this->WindowSize[1];
		this->ReleaseDepthPassGraphicsResources(ren->GetRenderWindow());
	}

	if (!this->DPFBO)
	{
		this->DPFBO = vtkOpenGLFramebufferObject::New();
	}

	this->DPFBO->SetContext(vtkOpenGLRenderWindow::SafeDownCast(
		ren->GetRenderWindow()));

	this->DPFBO->SaveCurrentBindingsAndBuffers();
	this->DPFBO->Bind(GL_FRAMEBUFFER);
	this->DPFBO->InitializeViewport(this->WindowSize[0], this->WindowSize[1]);

	if (!this->DPDepthBufferTextureObject ||
		!this->DPColorTextureObject)
	{
		this->DPDepthBufferTextureObject = vtkTextureObject::New();
		this->DPDepthBufferTextureObject->SetContext(
			vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));
		this->DPDepthBufferTextureObject->AllocateDepth(
			this->WindowSize[0], this->WindowSize[1], vtkTextureObject::Native);
		this->DPDepthBufferTextureObject->Activate();
		this->DPDepthBufferTextureObject->SetMinificationFilter(
			vtkTextureObject::Nearest);
		this->DPDepthBufferTextureObject->SetMagnificationFilter(
			vtkTextureObject::Nearest);
		this->DPDepthBufferTextureObject->SetAutoParameters(0);
		this->DPDepthBufferTextureObject->Bind();


		this->DPColorTextureObject = vtkTextureObject::New();

		this->DPColorTextureObject->SetContext(
			vtkOpenGLRenderWindow::SafeDownCast(
				ren->GetRenderWindow()));
		this->DPColorTextureObject->Create2D(this->WindowSize[0],
			this->WindowSize[1], 4,
			VTK_UNSIGNED_CHAR, false);
		this->DPColorTextureObject->Activate();
		this->DPColorTextureObject->SetMinificationFilter(
			vtkTextureObject::Nearest);
		this->DPColorTextureObject->SetMagnificationFilter(
			vtkTextureObject::Nearest);
		this->DPColorTextureObject->SetAutoParameters(0);

		this->DPFBO->AddDepthAttachment(
			GL_FRAMEBUFFER,
			this->DPDepthBufferTextureObject);

		this->DPFBO->AddColorAttachment(
			GL_FRAMEBUFFER, 0U,
			this->DPColorTextureObject);
	}


	this->DPFBO->ActivateDrawBuffers(1);
	this->DPFBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);

	// Setup the contour polydata mapper to render to DPFBO
	this->ContourMapper->SetInputConnection(
		this->ContourFilter->GetOutputPort());

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal::ExitDepthPass(
	vtkRenderer* vtkNotUsed(ren))
{
	this->DPFBO->DeactivateDrawBuffers();
	this->DPFBO->RestorePreviousBindingsAndBuffers();

	this->DPDepthBufferTextureObject->Deactivate();
	this->DPColorTextureObject->Deactivate();
	glDisable(GL_DEPTH_TEST);
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal
::ReleaseRenderToTextureGraphicsResources(vtkWindow* win)
{
	vtkOpenGLRenderWindow *rwin =
		vtkOpenGLRenderWindow::SafeDownCast(win);

	if (rwin)
	{
		if (this->FBO)
		{
			this->FBO->Delete();
			this->FBO = nullptr;
		}

		if (this->RTTDepthBufferTextureObject)
		{
			this->RTTDepthBufferTextureObject->ReleaseGraphicsResources(win);
			this->RTTDepthBufferTextureObject->Delete();
			this->RTTDepthBufferTextureObject = nullptr;
		}

		if (this->RTTDepthTextureObject)
		{
			this->RTTDepthTextureObject->ReleaseGraphicsResources(win);
			this->RTTDepthTextureObject->Delete();
			this->RTTDepthTextureObject = nullptr;
		}

		if (this->RTTColorTextureObject)
		{
			this->RTTColorTextureObject->ReleaseGraphicsResources(win);
			this->RTTColorTextureObject->Delete();
			this->RTTColorTextureObject = nullptr;
		}
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal
::ReleaseDepthPassGraphicsResources(vtkWindow* win)
{
	vtkOpenGLRenderWindow *rwin =
		vtkOpenGLRenderWindow::SafeDownCast(win);

	if (rwin)
	{
		if (this->DPFBO)
		{
			this->DPFBO->Delete();
			this->DPFBO = nullptr;
		}

		if (this->DPDepthBufferTextureObject)
		{
			this->DPDepthBufferTextureObject->ReleaseGraphicsResources(win);
			this->DPDepthBufferTextureObject->Delete();
			this->DPDepthBufferTextureObject = nullptr;
		}

		if (this->DPColorTextureObject)
		{
			this->DPColorTextureObject->ReleaseGraphicsResources(win);
			this->DPColorTextureObject->Delete();
			this->DPColorTextureObject = nullptr;
		}

		this->ContourMapper->ReleaseGraphicsResources(win);
	}
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::vtkInternal
::ReleaseImageSampleGraphicsResources(vtkWindow* win)
{
	vtkOpenGLRenderWindow *rwin =
		vtkOpenGLRenderWindow::SafeDownCast(win);

	if (rwin)
	{
		if (this->ImageSampleFBO)
		{
			this->ImageSampleFBO->Delete();
			this->ImageSampleFBO = nullptr;
		}

		for (auto& tex : this->ImageSampleTexture)
		{
			tex->ReleaseGraphicsResources(win);
			tex = nullptr;
		}
		this->ImageSampleTexture.clear();
		this->ImageSampleTexNames.clear();

		if (this->ImageSampleVBO)
		{
			this->ImageSampleVBO->Delete();
			this->ImageSampleVBO = nullptr;
		}

		if (this->ImageSampleVAO)
		{
			this->ImageSampleVAO->Delete();
			this->ImageSampleVAO = nullptr;
		}

		// Do not delete the shader program - Let the cache clean it up.
		this->ImageSampleProg = nullptr;
	}
}



//----------------------------------------------------------------------------
vtkLIC3DMapper::vtkLIC3DMapper() :
	vtkOpenGLGPUVolumeRayCastMapper()
{
	this->Impl = new vtkInternal(this);

	this->NoiseTextureSize[0] = this->NoiseTextureSize[1] = -1;
	this->LICNoiseSize[0] = this->LICNoiseSize[1] = this->LICNoiseSize[2] = 128;
	this->NoiseGenerator = nullptr;
	this->LICNoiseGenerator = nullptr;

	this->VolumeTexture = vtkVolumeTexture::New();
	this->VolumeTexture->SetMapper(this);

	this->VectorVolumeTex = vtkTextureObject::New();
}

//----------------------------------------------------------------------------
vtkLIC3DMapper::~vtkLIC3DMapper()
{
	if (this->NoiseGenerator)
	{
		this->NoiseGenerator->Delete();
		this->NoiseGenerator = nullptr;
	}

	delete this->Impl;
	this->Impl = nullptr;

	this->VolumeTexture->Delete();
	this->VolumeTexture = nullptr;

	this->VectorVolumeTex->Delete();
	this->VectorVolumeTex = nullptr;
}
//----------------------------------------------------------------------------
void vtkLIC3DMapper::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os, indent);
}

void vtkLIC3DMapper::SetNumberOfForwardSteps(int val)
{
	this->StepForward = val;
}

void vtkLIC3DMapper::SetNumberOfBackwardSteps(int val)
{
	this->StepBackward = val;
}

void vtkLIC3DMapper::SetLICStepSize(double val)
{
	this->LICStepSize = val;
}

void vtkLIC3DMapper::BuildShader(vtkRenderer* ren, vtkVolume* vol, int noOfCmponents)
{
	std::string vertexShader(LICraycastervs);
	std::string fragmentShader(LICraycasterfs);

	std::string LICfragmentShader(LICcomputefs);
	//fragmentShader.append(LICfragmentShader);

	this->ReplaceShaderRenderPass(vertexShader, fragmentShader, vol, false);

	// Now compile the shader
	//--------------------------------------------------------------------------
	this->Impl->ShaderProgram = this->Impl->ShaderCache->ReadyShaderProgram(
		vertexShader.c_str(), fragmentShader.c_str(), "");
	if (!this->Impl->ShaderProgram || !this->Impl->ShaderProgram->GetCompiled())
	{
		vtkErrorMacro("Shader failed to compile");
	}

	this->Impl->ShaderBuildTime.Modified();
}


//----------------------------------------------------------------------------
void vtkLIC3DMapper::GPURender(vtkRenderer* ren,
	vtkVolume* vol)
{
	vtkOpenGLClearErrorMacro();

	this->ResourceCallback->RegisterGraphicsResources(
		static_cast<vtkOpenGLRenderWindow *>(ren->GetRenderWindow()));

	this->Impl->TempMatrix1->Identity();

	this->Impl->NeedToInitializeResources =
		(this->Impl->ReleaseResourcesTime.GetMTime() >
			this->Impl->InitializationTime.GetMTime());

	// Make sure the context is current
	vtkOpenGLRenderWindow* renWin =
		vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
	renWin->MakeCurrent();

	// Update in_volume first to make sure states are current
	vol->Update();
	vtkImageData* input = this->GetTransformedInput();

	// vtkVolume ensures the property will be valid
	vtkVolumeProperty* volumeProperty = vol->GetProperty();
	vtkOpenGLCamera* cam = vtkOpenGLCamera::SafeDownCast(ren->GetActiveCamera());

	// Check whether we have independent components or not
	int const independentComponents = volumeProperty->GetIndependentComponents();

	this->Impl->CheckPropertyKeys(vol);

	// Get window size and corners
	if (!this->Impl->PreserveViewport)
	{
		ren->GetTiledSizeAndOrigin(
			this->Impl->WindowSize, this->Impl->WindowSize + 1,
			this->Impl->WindowLowerLeft, this->Impl->WindowLowerLeft + 1);
	}
	else
	{
		int vp[4];
		glGetIntegerv(GL_VIEWPORT, vp);
		this->Impl->WindowLowerLeft[0] = vp[0];
		this->Impl->WindowLowerLeft[1] = vp[1];
		this->Impl->WindowSize[0] = vp[2];
		this->Impl->WindowSize[1] = vp[3];
	}
	//input->SetScalarType(VTK_FLOAT);
	//input->AllocateScalars(VTK_FLOAT, 3);
	//vtkSmartPointer<vtkImageCast> castFilter =
	//	vtkSmartPointer<vtkImageCast>::New();
	//castFilter->SetInputData(input);
	//castFilter->SetOutputScalarTypeToFloat(); 
	//vtkSmartPointer<vtkImageData> floatImage = vtkSmartPointer<vtkImageData>::New();
	//floatImage->AllocateScalars(VTK_FLOAT, 3);
	////castFilter->SetOutput(floatImage);
	//castFilter->Update();
	//floatImage = castFilter->GetOutputDataObject();
	
	vtkDataArray* scalars = this->GetScalars(input,
		this->ScalarMode,
		this->ArrayAccessMode,
		this->ArrayId,
		this->ArrayName,
		this->CellFlag);

	if (strcmp(this->currentInputName, this->ArrayName) != 0)
	{
		strcpy(this->currentInputName, this->ArrayName);
		this->Impl->NeedToInitializeResources = true;
	}
	// Allocate important variables
	int const noOfComponents = scalars->GetNumberOfComponents();
	this->Impl->Bias.resize(noOfComponents, 0.0);

	if (this->Impl->NeedToInitializeResources ||
		(volumeProperty->GetMTime() > this->Impl->InitializationTime.GetMTime()))
	{
		this->Impl->InitializeTransferFunction(ren, vol, noOfComponents,
			independentComponents);
	}

	// Three dependent components are not supported
	if ((noOfComponents == 3) && !independentComponents)
	{
		vtkErrorMacro("Three dependent components are not supported");
	}

	// Update the volume if needed
	if (this->Impl->NeedToInitializeResources ||
		(input->GetMTime() > this->Impl->InputUpdateTime.GetMTime()))
	{
		this->Impl->LoadData(ren, vol, volumeProperty, input, scalars);
		this->Impl->LoadVectorVolumeTexture(ren, input, scalars);
	}
	else
	{
		this->Impl->LoadMask(ren, input, this->MaskInput, vol);
		this->Impl->UpdateVolume(volumeProperty);
	}

	this->ComputeReductionFactor(vol->GetAllocatedRenderTime());
	this->Impl->UpdateSamplingDistance(input, ren, vol);
	this->Impl->UpdateTransferFunction(ren, vol, noOfComponents,
		independentComponents);

	// Update noise sampler texture
	if (this->UseJittering)
	{
		this->Impl->CreateNoiseTexture(ren);
	}

	// Create Noise texture for 3D LIC computation
	this->Impl->CreateLICNoiseTexture(ren);
	this->Impl->CreateKernelFilterTexture(ren);

	// Grab depth sampler buffer (to handle cases when we are rendering geometry
	// and in_volume together
	this->Impl->CaptureDepthTexture(ren, vol);

	this->Impl->ShaderCache = vtkOpenGLRenderWindow::SafeDownCast(
		ren->GetRenderWindow())->GetShaderCache();

	this->Impl->CheckPickingState(ren);

	vtkMTimeType renderPassTime = this->GetRenderPassStageMTime(vol);

	if (this->UseDepthPass && this->GetBlendMode() ==
		vtkVolumeMapper::COMPOSITE_BLEND)
	{
		this->CurrentPass = DepthPass;

		if (this->Impl->NeedToInitializeResources ||
			volumeProperty->GetMTime() > this->Impl->DepthPassSetupTime.GetMTime() ||
			this->GetMTime() > this->Impl->DepthPassSetupTime.GetMTime() ||
			cam->GetParallelProjection() != this->Impl->LastProjectionParallel ||
			this->Impl->SelectionStateTime.GetMTime() > this->Impl->ShaderBuildTime.GetMTime() ||
			renderPassTime > this->Impl->ShaderBuildTime)
		{
			this->Impl->LastProjectionParallel =
				cam->GetParallelProjection();

			this->Impl->ContourFilter->SetInputData(input);
			for (int i = 0; i < this->GetDepthPassContourValues()->GetNumberOfContours(); ++i)
			{
				this->Impl->ContourFilter->SetValue(i,
					this->DepthPassContourValues->GetValue(i));
			}

			vtkNew<vtkMatrix4x4> newMatrix;
			newMatrix->DeepCopy(vol->GetMatrix());

			this->Impl->SetupDepthPass(ren);

			this->Impl->ContourActor->Render(ren,
				this->Impl->ContourMapper);

			this->Impl->ExitDepthPass(ren);

			this->Impl->DepthPassSetupTime.Modified();
			this->Impl->DepthPassTime.Modified();

			this->CurrentPass = RenderPass;
			this->BuildShader(ren, vol, noOfComponents);
		}
		else if (cam->GetMTime() > this->Impl->DepthPassTime.GetMTime())
		{
			this->Impl->SetupDepthPass(ren);

			this->Impl->ContourActor->Render(ren,
				this->Impl->ContourMapper);

			this->Impl->ExitDepthPass(ren);
			this->Impl->DepthPassTime.Modified();

			this->CurrentPass = RenderPass;
		}

		// Configure picking begin (changes blending, so needs to be called before
		// vtkVolumeStateRAII)
		if (this->Impl->IsPicking)
		{
			this->Impl->BeginPicking(ren);
		}

		// Set OpenGL states
		vtkVolumeStateRAII glState(this->Impl->PreserveGLState);

		if (this->RenderToImage)
		{
			this->Impl->SetupRenderToTexture(ren);
		}

		if (!this->Impl->PreserveViewport)
		{
			// NOTE: This is a must call or else, multiple viewport
			// rendering would not work. We need this primarily because
			// FBO set it otherwise.
			// TODO The viewport should not be set within the mapper,
			// causes issues when vtkOpenGLRenderPass instances modify it too.
			glViewport(this->Impl->WindowLowerLeft[0],
				this->Impl->WindowLowerLeft[1],
				this->Impl->WindowSize[0],
				this->Impl->WindowSize[1]);
		}

		renWin->GetShaderCache()->ReadyShaderProgram(this->Impl->ShaderProgram);

		this->Impl->DPDepthBufferTextureObject->Activate();
		this->Impl->ShaderProgram->SetUniformi("in_depthPassSampler",
			this->Impl->DPDepthBufferTextureObject->GetTextureUnit());

		this->DoGPURender(ren, vol, cam, this->Impl->ShaderProgram,
			noOfComponents, independentComponents);

		this->Impl->DPDepthBufferTextureObject->Deactivate();
	}
	else
	{
		// Configure picking begin (changes blending, so needs to be called before
		// vtkVolumeStateRAII)
		if (this->Impl->IsPicking)
		{
			this->Impl->BeginPicking(ren);
		}
		// Set OpenGL states
		vtkVolumeStateRAII glState(this->Impl->PreserveGLState);

		// Build shader now
		// First get the shader cache from the render window. This is important
		// to make sure that shader cache knows the state of various shader programs
		// in use.
		if (this->Impl->NeedToInitializeResources ||
			volumeProperty->GetMTime() > this->Impl->ShaderBuildTime.GetMTime() ||
			this->GetMTime() > this->Impl->ShaderBuildTime.GetMTime() ||
			cam->GetParallelProjection() != this->Impl->LastProjectionParallel ||
			this->Impl->SelectionStateTime.GetMTime() > this->Impl->ShaderBuildTime.GetMTime() ||
			renderPassTime > this->Impl->ShaderBuildTime)
		{
			this->Impl->LastProjectionParallel =
				cam->GetParallelProjection();
			this->BuildShader(ren, vol, noOfComponents);
		}
		else
		{
			// Bind the shader
			this->Impl->ShaderCache->ReadyShaderProgram(
				this->Impl->ShaderProgram);
		}

		if (this->RenderToImage)
		{
			this->Impl->SetupRenderToTexture(ren);


			this->DoGPURender(ren, vol, cam, this->Impl->ShaderProgram,
				noOfComponents, independentComponents);

			this->Impl->ExitRenderToTexture(ren);
		}
		else
		{
			this->Impl->BeginImageSample(ren, vol);
			this->DoGPURender(ren, vol, cam, this->Impl->ShaderProgram,
				noOfComponents, independentComponents);
			this->Impl->EndImageSample(ren);
		}
	}

	// Configure picking end
	if (this->Impl->IsPicking)
	{
		this->Impl->EndPicking(ren);
	}

	glFinish();
}

//----------------------------------------------------------------------------
void vtkLIC3DMapper::DoGPURender(vtkRenderer* ren,
	vtkVolume* vol,
	vtkOpenGLCamera* cam,
	vtkShaderProgram* prog,
	int noOfComponents,
	int independentComponents)
{
	// if the shader didn't compile return
	if (!prog)
	{
		return;
	}

	// Cell spacing is required to be computed globally (full volume extents)
	// given that gradients are computed globally (not per block).
	float fvalue3[3]; /* temporary value container */
	vtkInternal::ToFloat(this->Impl->CellSpacing, fvalue3);
	prog->SetUniform3fv("in_cellSpacing", 1, &fvalue3);

	this->SetShaderParametersRenderPass(vol);

	// Sort blocks in case the viewpoint changed, it immediately returns if there
	// is a single block.
	this->VolumeTexture->SortBlocksBackToFront(ren, vol->GetMatrix());

	vtkVolumeTexture::VolumeBlock* block = this->VolumeTexture->GetNextBlock();
	vtkVolumeTexture::VolumeBlock* maskBlock = nullptr;
	if (this->Impl->CurrentMask)
	{
		this->Impl->CurrentMask->SortBlocksBackToFront(ren, vol->GetMatrix());
		maskBlock = this->Impl->CurrentMask->GetNextBlock();
	}

	while (block != nullptr)
	{
		this->Impl->ComputeBounds(block->ImageData);

		// Cell step/scale are adjusted per block.
		// Step should be dependent on the bounds and not on the texture size
		// since we can have a non-uniform voxel size / spacing / aspect ratio.
		this->Impl->CellStep[0] =
			(1.0 / static_cast<double>(this->Impl->Extents[1] - this->Impl->Extents[0]));
		this->Impl->CellStep[1] =
			(1.0 / static_cast<double>(this->Impl->Extents[3] - this->Impl->Extents[2]));
		this->Impl->CellStep[2] =
			(1.0 / static_cast<double>(this->Impl->Extents[5] - this->Impl->Extents[4]));

		this->Impl->CellScale[0] = (this->Impl->LoadedBounds[1] -
			this->Impl->LoadedBounds[0]) * 0.5;
		this->Impl->CellScale[1] = (this->Impl->LoadedBounds[3] -
			this->Impl->LoadedBounds[2]) * 0.5;
		this->Impl->CellScale[2] = (this->Impl->LoadedBounds[5] -
			this->Impl->LoadedBounds[4]) * 0.5;

		vtkInternal::ToFloat(this->Impl->CellStep, fvalue3);
		prog->SetUniform3fv("in_cellStep", 1, &fvalue3);
		vtkInternal::ToFloat(this->Impl->CellScale, fvalue3);
		prog->SetUniform3fv("in_cellScale", 1, &fvalue3);

		// Update sampling distance
		this->Impl->DatasetStepSize[0] = 1.0 / (this->Impl->LoadedBounds[1] -
			this->Impl->LoadedBounds[0]);
		this->Impl->DatasetStepSize[1] = 1.0 / (this->Impl->LoadedBounds[3] -
			this->Impl->LoadedBounds[2]);
		this->Impl->DatasetStepSize[2] = 1.0 / (this->Impl->LoadedBounds[5] -
			this->Impl->LoadedBounds[4]);

		// Compute texture to dataset matrix
		this->Impl->TextureToDataSetMat->Identity();
		this->Impl->TextureToDataSetMat->SetElement(0, 0,
			(1.0 / this->Impl->DatasetStepSize[0]));
		this->Impl->TextureToDataSetMat->SetElement(1, 1,
			(1.0 / this->Impl->DatasetStepSize[1]));
		this->Impl->TextureToDataSetMat->SetElement(2, 2,
			(1.0 / this->Impl->DatasetStepSize[2]));
		this->Impl->TextureToDataSetMat->SetElement(3, 3,
			1.0);
		this->Impl->TextureToDataSetMat->SetElement(0, 3,
			this->Impl->LoadedBounds[0]);
		this->Impl->TextureToDataSetMat->SetElement(1, 3,
			this->Impl->LoadedBounds[2]);
		this->Impl->TextureToDataSetMat->SetElement(2, 3,
			this->Impl->LoadedBounds[4]);

		// Activate/bind DepthTextureObject to a texture unit first as it was already
		// activated in CaptureDepthTexture. Certain APPLE implementations seem to be
		// sensitive to swaping the activation order (causing GL_INVALID_OPERATION after
		// the glDraw call).
#if GL_ES_VERSION_3_0 != 1
		// currently broken on ES
		this->Impl->DepthTextureObject->Activate();
		prog->SetUniformi("in_depthSampler",
			this->Impl->DepthTextureObject->GetTextureUnit());
#endif

		// Bind current volume texture
		block->TextureObject->Activate();
		prog->SetUniformi("in_volume", block->TextureObject->GetTextureUnit());

		// Temporary variables
		float fvalue2[2];
		float fvalue4[4];

		vtkVolumeProperty* volumeProperty = vol->GetProperty();

		// Bind textures
		//--------------------------------------------------------------------------
		// Opacity, color, and gradient opacity samplers / textures
		int const numberOfSamplers = (independentComponents ? noOfComponents : 1);
		this->Impl->ActivateTransferFunction(prog, volumeProperty, numberOfSamplers);

		// simple volume data send to GPU
		if (this->VectorVolumeTex)
		{
			this->VectorVolumeTex->Activate();
			prog->SetUniformi("in_vector", this->VectorVolumeTex->GetTextureUnit());
		}

		if (this->Impl->NoiseTextureObject)
		{
			this->Impl->NoiseTextureObject->Activate();
			prog->SetUniformi("in_noiseSampler",
				this->Impl->NoiseTextureObject->GetTextureUnit());
		}

		if (this->Impl->LICNoiseTextureObject)
		{
			this->Impl->LICNoiseTextureObject->Activate();
			prog->SetUniformi("in_LICnoiseSampler",
				this->Impl->LICNoiseTextureObject->GetTextureUnit());
		}

		if (this->Impl->KernelFilterTextureObject)
		{
			this->Impl->KernelFilterTextureObject->Activate();
			prog->SetUniformi("licKernelSampler",
				this->Impl->KernelFilterTextureObject->GetTextureUnit());
		}

		float gradient[3] = { GradientScale, IllumScale, FreqScale };


		float invFilterSize = 1.0 / (this->StepForward + this->StepBackward);
		float licParams[3] = { this->StepForward, this->StepBackward, this->LICStepSize };
		float licKernel[3] = { 0.5f / this->StepForward, 0.5f / this->StepBackward, invFilterSize };
		float RaycastStepSize = this->VolumeStepScale;
		prog->SetUniform3fv("gradient", 1, &gradient);
		prog->SetUniform3fv("licParams", 1, &licParams);
		prog->SetUniform3fv("licKernel", 1, &licKernel);
		prog->SetUniform1fv("volumeStepScale", 1, &RaycastStepSize);


		if (maskBlock)
		{
			maskBlock->TextureObject->Activate();
			prog->SetUniformi("in_mask", maskBlock->TextureObject->GetTextureUnit());
		}

		if (noOfComponents == 1 &&
			this->BlendMode != vtkGPUVolumeRayCastMapper::ADDITIVE_BLEND)
		{
			if (this->MaskInput != nullptr && this->MaskType == LabelMapMaskType)
			{
				this->Impl->Mask1RGBTable->Activate();
				prog->SetUniformi("in_mask1",
					this->Impl->Mask1RGBTable->GetTextureUnit());

				this->Impl->Mask2RGBTable->Activate();
				prog->SetUniformi("in_mask2", this->Impl->Mask2RGBTable->GetTextureUnit());
				prog->SetUniformf("in_maskBlendFactor", this->MaskBlendFactor);
			}
		}

		// Bind light and material properties
		//--------------------------------------------------------------------------
		this->Impl->SetLightingParameters(ren, prog, vol);

		float ambient[4][3];
		float diffuse[4][3];
		float specular[4][3];
		float specularPower[4];

		for (int i = 0; i < numberOfSamplers; ++i)
		{
			ambient[i][0] = ambient[i][1] = ambient[i][2] =
				volumeProperty->GetAmbient(i);
			diffuse[i][0] = diffuse[i][1] = diffuse[i][2] =
				volumeProperty->GetDiffuse(i);
			specular[i][0] = specular[i][1] = specular[i][2] =
				volumeProperty->GetSpecular(i);
			specularPower[i] = volumeProperty->GetSpecularPower(i);
		}

		prog->SetUniform3fv("in_ambient", numberOfSamplers, ambient);
		prog->SetUniform3fv("in_diffuse", numberOfSamplers, diffuse);
		prog->SetUniform3fv("in_specular", numberOfSamplers, specular);
		prog->SetUniform1fv("in_shininess", numberOfSamplers, specularPower);

		// Bind matrices
		//--------------------------------------------------------------------------
		vtkMatrix4x4* glTransformMatrix;
		vtkMatrix4x4* modelViewMatrix;
		vtkMatrix3x3* normalMatrix;
		vtkMatrix4x4* projectionMatrix;
		cam->GetKeyMatrices(ren, modelViewMatrix, normalMatrix,
			projectionMatrix, glTransformMatrix);

		this->Impl->InverseProjectionMat->DeepCopy(projectionMatrix);
		this->Impl->InverseProjectionMat->Invert();
		prog->SetUniformMatrix("in_projectionMatrix", projectionMatrix);
		prog->SetUniformMatrix("in_inverseProjectionMatrix",
			this->Impl->InverseProjectionMat);

		this->Impl->InverseModelViewMat->DeepCopy(modelViewMatrix);
		this->Impl->InverseModelViewMat->Invert();
		prog->SetUniformMatrix("in_modelViewMatrix", modelViewMatrix);
		prog->SetUniformMatrix("in_inverseModelViewMatrix",
			this->Impl->InverseModelViewMat);

		this->Impl->TempMatrix1->DeepCopy(vol->GetMatrix());
		this->Impl->TempMatrix1->Transpose();
		this->Impl->InverseVolumeMat->DeepCopy(this->Impl->TempMatrix1);
		this->Impl->InverseVolumeMat->Invert();
		prog->SetUniformMatrix("in_volumeMatrix",
			this->Impl->TempMatrix1);
		prog->SetUniformMatrix("in_inverseVolumeMatrix",
			this->Impl->InverseVolumeMat);

		this->Impl->TempMatrix1->DeepCopy(this->Impl->TextureToDataSetMat);

		vtkMatrix4x4::Multiply4x4(vol->GetMatrix(),
			this->Impl->TempMatrix1,
			this->Impl->TextureToEyeTransposeInverse);

		vtkMatrix4x4::Multiply4x4(modelViewMatrix,
			this->Impl->TextureToEyeTransposeInverse,
			this->Impl->TextureToEyeTransposeInverse);

		this->Impl->TempMatrix1->Transpose();
		this->Impl->InverseTextureToDataSetMat->DeepCopy(
			this->Impl->TempMatrix1);
		this->Impl->InverseTextureToDataSetMat->Invert();

		prog->SetUniformMatrix("in_textureDatasetMatrix",
			this->Impl->TempMatrix1);
		prog->SetUniformMatrix("in_inverseTextureDatasetMatrix",
			this->Impl->InverseTextureToDataSetMat);
		prog->SetUniformMatrix("in_textureToEye",
			this->Impl->TextureToEyeTransposeInverse);

		// Bind other misc parameters
		//--------------------------------------------------------------------------
		if (cam->GetParallelProjection())
		{
			double dir[4];
			cam->GetDirectionOfProjection(dir);
			vtkInternal::ToFloat(dir[0], dir[1], dir[2], fvalue3);
			prog->SetUniform3fv(
				"in_projectionDirection", 1, &fvalue3);
		}

		// Pass constant uniforms at initialization
		prog->SetUniformi("in_noOfComponents", noOfComponents);
		prog->SetUniformi("in_independentComponents", independentComponents);

		// LargeDataTypes have been already biased and scaled so in those cases 0s
		// and 1s are passed respectively.
		float tscale[4] = { 1.0, 1.0, 1.0, 1.0 };
		float tbias[4] = { 0.0, 0.0, 0.0, 0.0 };
		float(*scalePtr)[4] = &tscale;
		float(*biasPtr)[4] = &tbias;
		//if (!this->VolumeTexture->HandleLargeDataTypes &&
		//	(noOfComponents == 1 || noOfComponents == 2 || independentComponents))
		//{
			scalePtr = &this->VolumeTexture->Scale;
			biasPtr = &this->VolumeTexture->Bias;
		//}
		prog->SetUniform4fv("in_volume_scale", 1, scalePtr);
		prog->SetUniform4fv("in_volume_bias", 1, biasPtr);

		prog->SetUniformf("in_sampleDistance", this->Impl->ActualSampleDistance);

		float scalarsRange[4][2];
		vtkInternal::ToFloat(this->VolumeTexture->ScalarRange, scalarsRange);
		prog->SetUniform2fv("in_scalarsRange", 4, scalarsRange);

		vtkInternal::ToFloat(cam->GetPosition(), fvalue3, 3);
		prog->SetUniform3fv("in_cameraPos", 1, &fvalue3);

		vtkInternal::ToFloat(this->Impl->LoadedBounds[0],
			this->Impl->LoadedBounds[2],
			this->Impl->LoadedBounds[4], fvalue3);
		prog->SetUniform3fv("in_volumeExtentsMin", 1, &fvalue3);

		vtkInternal::ToFloat(this->Impl->LoadedBounds[1],
			this->Impl->LoadedBounds[3],
			this->Impl->LoadedBounds[5], fvalue3);
		prog->SetUniform3fv("in_volumeExtentsMax", 1, &fvalue3);

		vtkInternal::ToFloat(this->Impl->Extents[0],
			this->Impl->Extents[2],
			this->Impl->Extents[4], fvalue3);
		prog->SetUniform3fv("in_textureExtentsMin", 1, &fvalue3);

		vtkInternal::ToFloat(this->Impl->Extents[1],
			this->Impl->Extents[3],
			this->Impl->Extents[5], fvalue3);
		prog->SetUniform3fv("in_textureExtentsMax", 1, &fvalue3);

		// TODO Take consideration of reduction factor
		vtkInternal::ToFloat(this->Impl->WindowLowerLeft, fvalue2);
		prog->SetUniform2fv("in_windowLowerLeftCorner", 1, &fvalue2);

		vtkInternal::ToFloat(1.0 / this->Impl->WindowSize[0],
			1.0 / this->Impl->WindowSize[1], fvalue2);
		prog->SetUniform2fv("in_inverseOriginalWindowSize", 1, &fvalue2);

		vtkInternal::ToFloat(1.0 / this->Impl->WindowSize[0],
			1.0 / this->Impl->WindowSize[1], fvalue2);
		prog->SetUniform2fv("in_inverseWindowSize", 1, &fvalue2);

		prog->SetUniformi("in_useJittering", this->GetUseJittering());

		prog->SetUniformi("in_cellFlag", this->CellFlag);
		vtkInternal::ToFloat(this->Impl->AdjustedTexMin[0],
			this->Impl->AdjustedTexMin[1],
			this->Impl->AdjustedTexMin[2], fvalue3);
		prog->SetUniform3fv("in_texMin", 1, &fvalue3);

		vtkInternal::ToFloat(this->Impl->AdjustedTexMax[0],
			this->Impl->AdjustedTexMax[1],
			this->Impl->AdjustedTexMax[2], fvalue3);
		prog->SetUniform3fv("in_texMax", 1, &fvalue3);

		this->Impl->TempMatrix1->DeepCopy(this->Impl->CellToPointMatrix);
		this->Impl->TempMatrix1->Transpose();
		prog->SetUniformMatrix("in_cellToPoint", this->Impl->TempMatrix1);

		prog->SetUniformi("in_clampDepthToBackface", this->GetClampDepthToBackface());

		// Bind cropping
		//--------------------------------------------------------------------------
		this->Impl->SetCroppingRegions(ren, prog, vol);

		// Bind clipping
		//--------------------------------------------------------------------------
		this->Impl->SetClippingPlanes(ren, prog, vol);

		// Bind the prop Id
		//--------------------------------------------------------------------------
		if (this->Impl->CurrentSelectionPass < vtkHardwareSelector::ID_LOW24)
		{
			this->Impl->SetPickingId(ren);
		}

		// Set the scalar range to be considered for average ip blend
		//--------------------------------------------------------------------------
		double avgRange[2];
		this->GetAverageIPScalarRange(avgRange);
		if (avgRange[1] < avgRange[0])
		{
			double tmp = avgRange[1];
			avgRange[1] = avgRange[0];
			avgRange[0] = tmp;
		}
		vtkInternal::ToFloat(avgRange[0], avgRange[1], fvalue2);
		prog->SetUniform2fv("in_averageIPRange", 1, &fvalue2);

		// Finally set the scale and bias for color correction
		//--------------------------------------------------------------------------
		prog->SetUniformf("in_scale", 1.0 / this->FinalColorWindow);
		prog->SetUniformf("in_bias",
			(0.5 - (this->FinalColorLevel / this->FinalColorWindow)));

		if (noOfComponents > 1 && independentComponents)
		{
			for (int i = 0; i < noOfComponents; ++i)
			{
				fvalue4[i] = static_cast<float>(volumeProperty->GetComponentWeight(i));
			}
			prog->SetUniform4fv("in_componentWeight", 1, &fvalue4);
		}

		// Render volume geometry to trigger render
		//--------------------------------------------------------------------------
		this->Impl->RenderVolumeGeometry(ren, prog, vol);

		// Undo binds and de-activate buffers
		//--------------------------------------------------------------------------
		block->TextureObject->Deactivate();
		if (this->Impl->NoiseTextureObject)
		{
			this->Impl->NoiseTextureObject->Deactivate();
		}
#if GL_ES_VERSION_3_0 != 1
		if (this->Impl->DepthTextureObject)
		{
			this->Impl->DepthTextureObject->Deactivate();
		}
#endif

		this->Impl->DeactivateTransferFunction(volumeProperty, numberOfSamplers);

		if (maskBlock)
		{
			maskBlock->TextureObject->Deactivate();
		}

		if (noOfComponents == 1 &&
			this->BlendMode != vtkGPUVolumeRayCastMapper::ADDITIVE_BLEND)
		{
			if (this->MaskInput != nullptr && this->MaskType == LabelMapMaskType)
			{
				this->Impl->Mask1RGBTable->Deactivate();
				this->Impl->Mask2RGBTable->Deactivate();
			}
		}

		vtkOpenGLCheckErrorMacro("failed after Render");

		// Update next block to render
		//---------------------------------------------------------------------------
		block = this->VolumeTexture->GetNextBlock();
		if (this->Impl->CurrentMask)
		{
			maskBlock = this->Impl->CurrentMask->GetNextBlock();
		}
	}
}