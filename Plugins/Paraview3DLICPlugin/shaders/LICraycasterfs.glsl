//VTK::System::Dec
/*=========================================================================

  Program:   Visualization Toolkit
  Module:    raycasterfs.glsl

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

//////////////////////////////////////////////////////////////////////////////
///
/// Inputs
///
//////////////////////////////////////////////////////////////////////////////

/// 3D texture coordinates form vertex shader
varying vec3 ip_textureCoords;
varying vec3 ip_vertexPos;

//////////////////////////////////////////////////////////////////////////////
///
/// Outputs
///
//////////////////////////////////////////////////////////////////////////////

vec4 g_fragColor = vec4(0.0);

//////////////////////////////////////////////////////////////////////////////
///
/// Uniforms, attributes, and globals
///
//////////////////////////////////////////////////////////////////////////////
vec3 g_dataPos;
vec3 g_dirStep;
vec4 g_srcColor;
vec4 g_eyePosObj;
bool g_exit;
bool g_skip;
float g_currentT;
float g_terminatePointMax;

uniform vec4 in_volume_scale;
uniform vec4 in_volume_bias;

//VTK::Output::Dec

      
// Volume dataset      
uniform sampler3D in_volume;      
uniform int in_noOfComponents;      
uniform int in_independentComponents;      
      
uniform sampler2D in_noiseSampler;
uniform sampler3D in_LICnoiseSampler;  
uniform sampler1D licKernelSampler;    
#ifndef GL_ES      
uniform sampler2D in_depthSampler;      
#endif      
      
// Camera position      
uniform vec3 in_cameraPos;      
      
// view and model matrices      
uniform mat4 in_volumeMatrix;      
uniform mat4 in_inverseVolumeMatrix;      
uniform mat4 in_projectionMatrix;      
uniform mat4 in_inverseProjectionMatrix;      
uniform mat4 in_modelViewMatrix;      
uniform mat4 in_inverseModelViewMatrix;      
uniform mat4 in_textureDatasetMatrix;      
uniform mat4 in_inverseTextureDatasetMatrix;      
varying mat4 ip_inverseTextureDataAdjusted;      
uniform vec3 in_texMin;      
uniform vec3 in_texMax;      
uniform mat4 in_textureToEye;      
      
// Ray step size      
uniform vec3 in_cellStep;      
uniform vec2 in_scalarsRange[4];      
uniform vec3 in_cellSpacing;      
      
// Sample distance      
uniform float in_sampleDistance;      
      
// Scales      
uniform vec3 in_cellScale;      
uniform vec2 in_windowLowerLeftCorner;      
uniform vec2 in_inverseOriginalWindowSize;      
uniform vec2 in_inverseWindowSize;      
uniform vec3 in_textureExtentsMax;      
uniform vec3 in_textureExtentsMin;      
      
// Material and lighting      
uniform vec3 in_diffuse[4];      
uniform vec3 in_ambient[4];      
uniform vec3 in_specular[4];      
uniform float in_shininess[4];      
      
// Others      
uniform bool in_cellFlag;      
uniform bool in_useJittering;      
vec3 g_rayJitter = vec3(0.0);      
uniform bool in_clampDepthToBackface;      
      
uniform vec2 in_averageIPRange;        
uniform vec3 in_lightAmbientColor[1];        
uniform vec3 in_lightDiffuseColor[1];        
uniform vec3 in_lightSpecularColor[1];        
vec4 g_lightPosObj;        
vec3 g_ldir;        
vec3 g_vdir;        
vec3 g_h;

      
 const float g_opacityThreshold = 1.0 - 1.0 / 255.0;

//VTK::Cropping::Dec

      
 int clippingPlanesSize;      
 vec3 objRayDir;      
 mat4 textureToObjMat;

//VTK::Shading::Dec

//VTK::BinaryMask::Dec

//VTK::CompositeMask::Dec

//VTK::GradientCache::Dec

//3DLIC::LIC Parameter
uniform sampler3D in_vector;
uniform vec3 gradient; // scale, illum scale, frequency scale
uniform vec3 licParams;  // lics steps forward, backward, lic step width

uniform vec3 licKernel;  // kernel step width forward (0.5/licParams.x),
                         // kernel step width backward (0.5/licParams.y),
                         // inverse filter area

uniform float volumeStepScale;

vec3 frequency(in vec3 objPos, in float freqScale,
	out float logEyeDist)
{
	mat4 ModelViewProjectionMatrix = in_projectionMatrix * in_modelViewMatrix;
	float eyeDist = dot(vec4(ModelViewProjectionMatrix[0][2],
		ModelViewProjectionMatrix[1][2],
		ModelViewProjectionMatrix[2][2],
		ModelViewProjectionMatrix[3][2]),
		vec4(objPos, 1.0));
	logEyeDist = log2(eyeDist + 1.0);

	vec2 freqMeasure = vec2(-floor(logEyeDist));
	float freqRemainder = logEyeDist + freqMeasure.x;
	//  remainder is needed for
	//  LRP illum, freqMeasureFrac.w, noise2, noise1;

	freqMeasure.x = exp2(freqMeasure.x);
	freqMeasure.y = exp2(freqMeasure.y - 1.0);

	freqMeasure *= freqScale;

	return vec3(freqMeasure, freqRemainder);
}


vec4 noiseLookupGrad(in vec3 objPos,
	in float freqScale,
	out float logEyeDist)
{
	vec3 freqMeasure = frequency(objPos, freqScale, logEyeDist);

	vec3 freqTexCoord = objPos * freqMeasure.x;
	vec4 noise1 = texture3D(in_LICnoiseSampler, freqTexCoord);
	freqTexCoord = objPos * freqMeasure.y;
	vec4 noise2 = texture3D(in_LICnoiseSampler, freqTexCoord);

	// scale gradients from [0,1] to [-1,1]
	noise1.rgb = 2.0*noise1.rgb - 1.0;
	noise2.rgb = 2.0*noise2.rgb - 1.0;

	// perform linear interpolation
	return mix(noise1, noise2, freqMeasure.z);
}


float noiseLookup(in vec3 objPos, in float freqScale, out float logEyeDist)
{
	vec3 freqMeasure = frequency(objPos, freqScale, logEyeDist);

	vec3 freqTexCoord = objPos * freqMeasure.x;
	float noise1 = texture3D(in_LICnoiseSampler, freqTexCoord).r;
	freqTexCoord = objPos * freqMeasure.y;
	float noise2 = texture3D(in_LICnoiseSampler, freqTexCoord).r;

	// perform linear interpolation
	return mix(noise1, noise2, freqMeasure.z);
}


// preserve same spatial frequencies of the noise with respect to
// the image plane
vec4 freqSamplingGrad(in vec3 pos, out float logEyeDist)
{
	//vec3 objPos = pos * scaleVolInv.xyz;

	//vec4 tmp = noiseLookupGrad(pos, gradient.z, logEyeDist);
	return texture3D(in_LICnoiseSampler, pos);
	//return noiseLookupGrad(pos, gradient.z, logEyeDist);
}


float freqSampling(in vec3 pos, out float logEyeDist, in sampler3D vectorVolume = in_volume)
{
	//vec3 objPos = pos * scaleVolInv.xyz;

	//Use scalar data to decide noise range to be integrated
	vec4 vectorData = texture3D(vectorVolume, pos);

	//return texture3D(in_LICnoiseSampler, pos).r;
	return texture3D(in_LICnoiseSampler, pos*gradient.z).a;
	//return noiseLookup(pos, gradient.z, logEyeDist);
}


float singleLICstep(in vec3 licdir, in out vec3 newPos,
	in out vec4 step, in float kernelOffset,
	in out float logEyeDist, in float dir,
	in sampler3D vectorVolume = in_volume)
{
	float noise;

	// scale with LIC step size
	// also correct length according to camera distance
	licdir *= licParams.z * (logEyeDist*0.5 + 0.3);
	//vec3 Pos2 = newPos + licdir;
	//vec4 step2 = texture3D(vectorVolume, Pos2);
	//vec4 step2_ori = (step2 - in_volume_bias) / in_volume_scale;
	//vec3 licdir2 = normalize(step2_ori.rgb);
	//licdir2 *= dir;
	//licdir2 *= licParams.z * (logEyeDist*0.5 + 0.3);
	//Pos2 += 0.5 * licdir2;
	//newPos += 0.5 * (licdir + licdir2);
	newPos += licdir;

	step = texture3D(vectorVolume, newPos);
	vec4 step_ori = step * in_volume_scale;
	step = vec4(normalize(step_ori.rgb), 1.0);

	noise = freqSampling(newPos, logEyeDist);

	// determine weighting
	noise *= texture1D(licKernelSampler, kernelOffset).r;

	return noise;
}



// performs the LIC computation for n steps forward and backward
//    pos determines the starting position of the LIC
//    vectorFieldSample is the value of the vector field at this position
vec4 computeLIC(in vec3 pos, in vec4 vectorFieldSample,
	out vec2 streamDis, out vec3 streamStart, out vec3 streamEnd, in sampler3D vectorVolume = in_volume)
{
	vec3 licdir;
	float logEyeDist;
	float kernelOffset = 0.5;

	// perform first lookup
	float noise;
	float illum = freqSampling(pos, logEyeDist);

	// weight sample with lic kernel at position 0
	illum *= texture1D(licKernelSampler, 0.5).r;
	//return vec4(illum);
	float dir = -1;
	// backward LIC
	vec3 newPos = pos;
	vec4 step = vectorFieldSample;
	float streamlineL = 0.0;
	float sumCross = 0.0;
	for (int i = 0; i<int(licParams.y); ++i)
	{
		//licdir = -2.0*step.rgb + 1.0;
		licdir = step.rgb;
		kernelOffset -= licKernel.y;
		vec3 oldPos = newPos;
		illum += singleLICstep(licdir, newPos, step,
			kernelOffset, logEyeDist, dir, vectorVolume);
		streamlineL += length(newPos - oldPos);
		sumCross += length(cross(oldPos, newPos));
	}
	streamEnd = newPos;
	// forward LIC
	dir = 1;
	newPos = pos;
	step = vectorFieldSample;
	kernelOffset = 0.5;
	for (int i = 0; i<int(licParams.x); ++i)
	{
		//licdir = 2.0*step.rgb - 1.0;
		licdir = step.rgb;
		kernelOffset += licKernel.x;
		vec3 oldPos = newPos;
		illum += singleLICstep(licdir, newPos, step,
			kernelOffset, logEyeDist, dir, vectorVolume);
		streamlineL += length(newPos - oldPos);
		sumCross += length(cross(newPos, oldPos));
	}
	streamStart = newPos;
	streamDis = vec2(streamlineL, sumCross);
	return vec4(illum );
}


        
uniform sampler2D in_opacityTransferFunc;        
float computeOpacity(vec4 scalar)        
{        
  return texture2D(in_opacityTransferFunc, vec2(scalar.w, 0)).r;        
}

        
vec4 computeGradient(int component)        
  {        
  return vec4(0.0);        
  }

//VTK::ComputeGradientOpacity1D::Dec

      
vec4 computeLighting(vec4 color, int component)      
  {      
  vec4 finalColor = vec4(0.0);
  finalColor = vec4(color.rgb, 0.0);      
  finalColor.a = color.a;      
  return finalColor;      
  }

          
uniform sampler2D in_colorTransferFunc;          
vec4 computeColor(vec4 scalar, float opacity)          
  {          
  return computeLighting(vec4(texture2D(in_colorTransferFunc,          
                         vec2(scalar.w, 0.0)).xyz, opacity), 0);          
  }

        
vec3 computeRayDirection()        
  {        
  return normalize(ip_vertexPos.xyz - g_eyePosObj.xyz);        
  }


vec4 illumLIC(in float illum, in vec4 tfData)
{
	vec4 color;

	// result = lic intensity * color * illumination scaling
	color.rgb = illum * tfData.rgb * gradient.g;

	// alpha affected by lic intensity
	//float scale_illum = (illum - in_volume_bias.r) / in_volume_scale.r;
	color.a = texture2D(in_opacityTransferFunc, vec2(illum*1.3, 1)).r * tfData.a;
	//color.a = texture1D(transferAlphaOpacSampler, illum*1.3).a * tfData.a;

	// opacity and color correction
	float alphaCorrection = length(g_dirStep) * 128;
	color.a = 1.0 - pow(1.0 - color.a, alphaCorrection);

	return color;
}

//VTK::Picking::Dec

//VTK::RenderToImage::Dec

//VTK::DepthPeeling::Dec

/// We support only 8 clipping planes for now
/// The first value is the size of the data array for clipping
/// planes (origin, normal)
uniform float in_clippingPlanes[49];
uniform float in_scale;
uniform float in_bias;

//////////////////////////////////////////////////////////////////////////////
///
/// Helper functions
///
//////////////////////////////////////////////////////////////////////////////

/**
 * Transform window coordinate to NDC.
 */
vec4 WindowToNDC(const float xCoord, const float yCoord, const float zCoord)
{
  vec4 NDCCoord = vec4(0.0, 0.0, 0.0, 1.0);

  NDCCoord.x = (xCoord - in_windowLowerLeftCorner.x) * 2.0 *
    in_inverseWindowSize.x - 1.0;
  NDCCoord.y = (yCoord - in_windowLowerLeftCorner.y) * 2.0 *
    in_inverseWindowSize.y - 1.0;
  NDCCoord.z = (2.0 * zCoord - (gl_DepthRange.near + gl_DepthRange.far)) /
    gl_DepthRange.diff;

  return NDCCoord;
}

/**
 * Transform NDC coordinate to window coordinates.
 */
vec4 NDCToWindow(const float xNDC, const float yNDC, const float zNDC)
{
  vec4 WinCoord = vec4(0.0, 0.0, 0.0, 1.0);

  WinCoord.x = (xNDC + 1.f) / (2.f * in_inverseWindowSize.x) +
    in_windowLowerLeftCorner.x;
  WinCoord.y = (yNDC + 1.f) / (2.f * in_inverseWindowSize.y) +
    in_windowLowerLeftCorner.y;
  WinCoord.z = (zNDC * gl_DepthRange.diff +
    (gl_DepthRange.near + gl_DepthRange.far)) / 2.f;

  return WinCoord;
}

//////////////////////////////////////////////////////////////////////////////
///
/// Ray-casting
///
//////////////////////////////////////////////////////////////////////////////

/**
 * Global initialization. This method should only be called once per shader
 * invocation regardless of whether castRay() is called several times (e.g.
 * vtkDualDepthPeelingPass). Any castRay() specific initialization should be
 * placed within that function.
 */
void initializeRayCast()
{
  /// Initialize g_fragColor (output) to 0
  g_fragColor = vec4(0.0);
  g_dirStep = vec3(0.0);
  g_srcColor = vec4(0.0);
  g_exit = false;

        
  bool l_adjustTextureExtents =  !in_cellFlag;        
  // Get the 3D texture coordinates for lookup into the in_volume dataset        
  g_dataPos = ip_textureCoords.xyz;        
        
  // Eye position in dataset space        
  g_eyePosObj = (in_inverseVolumeMatrix * vec4(in_cameraPos, 1.0));        
  if (g_eyePosObj.w != 0.0)        
    {        
    g_eyePosObj.x /= g_eyePosObj.w;        
    g_eyePosObj.y /= g_eyePosObj.w;        
    g_eyePosObj.z /= g_eyePosObj.w;        
    g_eyePosObj.w = 1.0;        
    }        
        
  // Getting the ray marching direction (in dataset space);        
  vec3 rayDir = computeRayDirection();        
        
  // Multiply the raymarching direction with the step size to get the        
  // sub-step size we need to take at each raymarching step        
  g_dirStep = (ip_inverseTextureDataAdjusted *        
              vec4(rayDir, 0.0)).xyz * in_sampleDistance;        
        
  // 2D Texture fragment coordinates [0,1] from fragment coordinates.        
  // The frame buffer texture has the size of the plain buffer but         
  // we use a fraction of it. The texture coordinate is less than 1 if        
  // the reduction factor is less than 1.        
  // Device coordinates are between -1 and 1. We need texture        
  // coordinates between 0 and 1. The in_noiseSampler and in_depthSampler        
  // buffers have the original size buffer.        
  vec2 fragTexCoord = (gl_FragCoord.xy - in_windowLowerLeftCorner) *        
                      in_inverseWindowSize;        
        
  if (in_useJittering)        
  {        
    float jitterValue = texture2D(in_noiseSampler, fragTexCoord).x;        
    g_rayJitter = g_dirStep * jitterValue;        
    g_dataPos += g_rayJitter;        
  }        
  else        
  {        
    g_dataPos += g_dirStep;        
  }        
        
  // Flag to deternmine if voxel should be considered for the rendering        
  g_skip = false;

        
  // Flag to indicate if the raymarch loop should terminate       
  bool stop = false;      
      
  g_terminatePointMax = 0.0;      
      
#ifdef GL_ES      
  vec4 l_depthValue = vec4(1.0,1.0,1.0,1.0);      
#else      
  vec4 l_depthValue = texture2D(in_depthSampler, fragTexCoord);      
#endif      
  // Depth test      
  if(gl_FragCoord.z >= l_depthValue.x)      
    {      
    discard;      
    }      
      
  // color buffer or max scalar buffer have a reduced size.      
  fragTexCoord = (gl_FragCoord.xy - in_windowLowerLeftCorner) *      
                 in_inverseOriginalWindowSize;      
      
  // Compute max number of iterations it will take before we hit      
  // the termination point      
      
  // Abscissa of the point on the depth buffer along the ray.      
  // point in texture coordinates      
  vec4 terminatePoint = WindowToNDC(gl_FragCoord.x, gl_FragCoord.y, l_depthValue.x);      
      
  // From normalized device coordinates to eye coordinates.      
  // in_projectionMatrix is inversed because of way VT      
  // From eye coordinates to texture coordinates      
  terminatePoint = ip_inverseTextureDataAdjusted *      
                   in_inverseVolumeMatrix *      
                   in_inverseModelViewMatrix *      
                   in_inverseProjectionMatrix *      
                   terminatePoint;      
  terminatePoint /= terminatePoint.w;      
      
  g_terminatePointMax = length(terminatePoint.xyz - g_dataPos.xyz) /      
                        length(g_dirStep);      
  g_currentT = 0.0;

  //VTK::Cropping::Init

  //VTK::Clipping::Init

  //VTK::RenderToImage::Init

  //VTK::DepthPass::Init
}

/**
 * March along the ray direction sampling the volume texture.  This function
 * takes a start and end point as arguments but it is up to the specific render
 * pass implementation to use these values (e.g. vtkDualDepthPeelingPass). The
 * mapper does not use these values by default, instead it uses the number of
 * steps defined by g_terminatePointMax.
 */
vec4 castRay(const float zStart, const float zEnd)
{
	//VTK::DepthPeeling::Ray::Init

	//VTK::DepthPeeling::Ray::PathCheck

	//VTK::Shading::Init

	/// For all samples along the ray
	while (!g_exit)
	{

		g_skip = false;

		//VTK::Cropping::Impl

		//VTK::Clipping::Impl

		//VTK::BinaryMask::Impl

		//VTK::CompositeMask::Impl

		//VTK::PreComputeGradients::Impl


		if (!g_skip)
		{
			vec4 vectordata = texture3D(in_volume, g_dataPos); // 0 -1
			vec4 vec_orig = vectordata;

			vec3 streamStart;
			vec3 streamEnd;
			vec2 streamDis;
			//normalize vector data
			//vectordata = vec4(normalize(vectordata.rgb),length(vectordata.rgb));
			vectordata.rgb = normalize(vectordata.rgb);
			vectordata.a = 1.0;
			vec4 intensity = vec4(0.0);
			vec4 scalar = vec4(0.0);
			vec4 tfData = vec4(normalize(vectordata.rgb)*0.5 + vec3(0.5), 1.0);
			if (length(vectordata.rgb) >= 0.01)
			{
				intensity = computeLIC(g_dataPos, vectordata, streamDis, streamStart, streamEnd);
				intensity.a *= licKernel.b * gradient.r;
				//scalar *= 2;

				//scalar.r = scalar.r*in_volume_scale.r + in_volume_bias.r;     // map to 0-1  
				//scalar = normalize(scalar);

				//scalar = (intensity - in_volume_bias) / in_volume_scale;
				//scalar = vec4(scalar.r, scalar.g, scalar.b, scalar.r);
				//g_srcColor = intensity;
				             
				g_srcColor.a = computeOpacity(intensity);
				//g_srcColor.a = clamp(scalar.a, 0 , 1); 
				//tfData = computeColor(vec4(vec_orig.r), g_srcColor.a);
			}
			else
				g_srcColor = vec4(0.0);
			if (g_srcColor.a > 0)
			{   
				g_srcColor = illumLIC(intensity.a, tfData);

				//g_srcColor.a = computeOpacity(vec4(vec_orig.r));

				g_srcColor.rgb *= g_srcColor.a;
				g_fragColor = clamp((1.0f - g_fragColor.a) * g_srcColor + g_fragColor, 0.0, 1.0);
			}
		}

		//VTK::RenderToImage::Impl

		//VTK::DepthPass::Impl

		/// Advance ray
		g_dataPos += g_dirStep*volumeStepScale;


		if (any(greaterThan(g_dataPos, in_texMax)) ||
			any(lessThan(g_dataPos, in_texMin)))
		{
			break;
		}

		// Early ray termination      
		// if the currently composited colour alpha is already fully saturated      
		// we terminated the loop or if we have hit an obstacle in the      
		// direction of they ray (using depth buffer) we terminate as well.      
		if ((g_fragColor.a > g_opacityThreshold) ||
			g_currentT >= g_terminatePointMax)
		{
			break;
		}
		++g_currentT;
	}

	//VTK::Shading::Exit

	return g_fragColor;
}

/**
 * Finalize specific modes and set output data.
 */
void finalizeRayCast()
{
  //VTK::Base::Exit

  //VTK::Terminate::Exit

  //VTK::Cropping::Exit

  //VTK::Clipping::Exit

  //VTK::Picking::Exit

  g_fragColor.r = g_fragColor.r * in_scale + in_bias * g_fragColor.a;
  g_fragColor.g = g_fragColor.g * in_scale + in_bias * g_fragColor.a;
  g_fragColor.b = g_fragColor.b * in_scale + in_bias * g_fragColor.a;
  gl_FragData[0] = g_fragColor;

  //VTK::RenderToImage::Exit

  //VTK::DepthPass::Exit
}

//////////////////////////////////////////////////////////////////////////////
///
/// Main
///
//////////////////////////////////////////////////////////////////////////////
void main()
{
      
  initializeRayCast();    
  castRay(-1.0, -1.0);    
  finalizeRayCast();
}

