#ifndef vtkLIC3DMapper_h
#define vtkLIC3DMapper_h

#include "vtkNew.h"                          // For vtkNew
#include "vtkRenderingVolumeOpenGL2Module.h" // For export macro
#include "vtkOpenGLGPUVolumeRayCastMapper.h"


//----------------------------------------------------------------------------
class VTK_EXPORT vtkLIC3DMapper :
	public vtkOpenGLGPUVolumeRayCastMapper
{
public:
	static vtkLIC3DMapper* New();
	vtkTypeMacro(vtkLIC3DMapper, vtkOpenGLGPUVolumeRayCastMapper);
	void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;

	void SetLICStepSize(double val);
	void SetNumberOfForwardSteps(int val);
	void SetNumberOfBackwardSteps(int val);

	vtkSetMacro(GradientScale, double);
	vtkGetMacro(GradientScale, double);

	vtkSetMacro(IllumScale, double);
	vtkGetMacro(IllumScale, double);

	vtkSetMacro(FreqScale, double);
	vtkGetMacro(FreqScale, double);

	vtkSetMacro(VolumeStepScale, double);
	vtkGetMacro(VolumeStepScale, double);
protected:
	vtkLIC3DMapper();
	~vtkLIC3DMapper() override;

	// Description:
	// Build vertex and fragment shader for the volume rendering
	void BuildShader(vtkRenderer* ren, vtkVolume* vol, int noOfCmponents);
	
	// Description:
	// Rendering volume on GPU
	void GPURender(vtkRenderer *ren, vtkVolume *vol) override;

	// Description:
	// Method that performs the actual rendering given a volume and a shader
	void DoGPURender(vtkRenderer* ren,
		vtkVolume* vol,
		vtkOpenGLCamera* cam,
		vtkShaderProgram* shaderProgram,
		int noOfComponents,
		int independentComponents);
	int StepForward;
	int StepBackward;
	double LICStepSize;

	double GradientScale;
	double IllumScale;
	double FreqScale;

	double VolumeStepScale;

private:
	class vtkInternal;
	vtkInternal* Impl;

	friend class vtkVolumeTexture;
	vtkVolumeTexture* VolumeTexture;
	vtkTextureObject* VectorVolumeTex;
	char currentInputName[100];

	vtkImplicitFunction* NoiseGenerator;
	vtkImplicitFunction* LICNoiseGenerator;
	int NoiseTextureSize[2];
	int LICNoiseSize[3];

	vtkLIC3DMapper(const vtkLIC3DMapper&) = delete;
	void operator=(const vtkLIC3DMapper&) = delete;
};

#endif

