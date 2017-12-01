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
    float noise1 = texture3D(in_LICnoiseSampler, freqTexCoord).a;
    freqTexCoord = objPos * freqMeasure.y;
    float noise2 = texture3D(in_LICnoiseSampler, freqTexCoord).a;

    // perform linear interpolation
    return mix(noise1, noise2, freqMeasure.z);
}


// preserve same spatial frequencies of the noise with respect to
// the image plane
vec4 freqSamplingGrad(in vec3 pos, out float logEyeDist)
{
    //vec3 objPos = pos * scaleVolInv.xyz;

    //vec4 tmp = noiseLookupGrad(pos, gradient.z, logEyeDist);
   // return texture3D(in_LICnoiseSampler, pos);
    return noiseLookupGrad(pos, gradient.z, logEyeDist);
}


float freqSampling(in vec3 pos, out float logEyeDist, in sampler3D vectorVolume = in_volume)
{
    //vec3 objPos = pos * scaleVolInv.xyz;

	//Use scalar data to decide noise range to be integrated
	vec4 vectorData = texture3D(vectorVolume, pos);
	
	//return texture3D(in_LICnoiseSampler, pos).a
	return texture3D(in_LICnoiseSampler, pos*gradient.z).a;
	//return noiseLookup(pos, gradient.z, logEyeDist);
}


#ifdef USE_NOISE_GRADIENTS
vec4 singleLICstep(in vec3 licdir, in out vec3 newPos,
                   in out vec4 step, in float kernelOffset,
                   in out float logEyeDist, in float dir,
				   in sampler3D vectorVolume  = in_volume)
{
    vec4 noise;
#else
float singleLICstep(in vec3 licdir, in out vec3 newPos,
                   in out vec4 step, in float kernelOffset,
                   in out float logEyeDist, in float dir, 
				   in sampler3D vectorVolume  = in_volume)
{
    float noise;
#endif

    // consider speed of flow
#ifdef SPEED_OF_FLOW
    licdir *= step.a;
#endif

    // scale with LIC step size
    // also correct length according to camera distance
    licdir *= licParams.z * (logEyeDist*0.5 + 0.3);
    vec3 Pos2 = newPos + licdir;
	vec4 step2 = texture3D(vectorVolume, Pos2);
	vec3 licdir2 = 2.0*step2.rgb - 1.0;
	licdir2 *= dir;
	//licdir2 = step2.rgb;
#ifdef SPEED_OF_FLOW
    licdir2 *= step.a;
#endif
	licdir2 *= licParams.z * (logEyeDist*0.5 + 0.3);
	//Pos2 += 0.5 * licdir2;
	newPos += 0.5 * (licdir + licdir2);
	//newPos += 0.3 * licdir;

    step = texture3D(vectorVolume, newPos);
#ifdef TIME_DEPENDENT
    vectorFieldSample2 = texture3D(volumeSampler2, newPos);
    step = mix(timeStep, step, vectorFieldSample2);
#endif

#ifdef USE_NOISE_GRADIENTS
    noise = freqSamplingGrad(newPos, logEyeDist);
    //noise = vec4(texture3D(in_LICnoiseSampler, newPos).a);
#else
    noise = freqSampling(newPos, logEyeDist);
#endif

    // determine weighting
    noise *= texture1D(licKernelSampler, kernelOffset).r;

    return noise;
}



// performs the LIC computation for n steps forward and backward
//    pos determines the starting position of the LIC
//    vectorFieldSample is the value of the vector field at this position
vec4 computeLIC(in vec3 pos, in vec4 vectorFieldSample, out vec2 streamDis, out vec3 streamStart, out vec3 streaEnd, in sampler3D vectorVolume = in_volume)
{
    vec3 licdir;
    float logEyeDist;
    float kernelOffset = 0.5;
#ifdef TIME_DEPENDENT
    vec4 vectorFieldSample2;
#endif

    // perform first lookup

#ifdef USE_NOISE_GRADIENTS
    vec4 noise;
    vec4 illum = freqSamplingGrad(pos, logEyeDist, vectorVolume);
#else
    float noise;
    float illum = freqSampling(pos, logEyeDist);
#endif

    // weight sample with lic kernel at position 0
    illum *= texture1D(licKernelSampler, 0.5).r;
    
	float dir = -1;
    // backward LIC
    vec3 newPos = pos;
    vec4 step = vectorFieldSample;
	float streamlineL = 0.0;
	float sumCross = 0.0;
    for (int i=0; i<int(licParams.y); ++i)
    {
        licdir = -2.0*step.rgb + 1.0;
		//licdir = step.rgb;
        kernelOffset -= licKernel.y;
		vec3 oldPos = newPos;
        illum += singleLICstep(licdir, newPos, step, 
                               kernelOffset, logEyeDist, dir, vectorVolume);
		streamlineL += length(newPos - oldPos);
		sumCross += length(cross(oldPos, newPos));
    }
	streaEnd = newPos;
    // forward LIC
	dir = 1;
    newPos = pos;
    step = vectorFieldSample;
    kernelOffset = 0.5;
    for (int i=0; i<int(licParams.x); ++i)
    {
        licdir = 2.0*step.rgb - 1.0;
		//licdir = step.rgb;
        kernelOffset += licKernel.x;
		vec3 oldPos = newPos;
        illum += singleLICstep(licdir, newPos, step, 
                               kernelOffset, logEyeDist, dir, vectorVolume);
		streamlineL += length(newPos - oldPos);
		sumCross += length(cross(newPos, oldPos));
    }
    streamStart = newPos;
	streamDis = vec2(streamlineL, sumCross);
    return vec4(illum);
}


