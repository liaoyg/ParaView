//VTK::System::Dec

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    raycastervs.glsl

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/// Needed to enable inverse function
#extension GL_ARB_gpu_shader5 : enable

//////////////////////////////////////////////////////////////////////////////
///
/// Uniforms, attributes, and globals
///
//////////////////////////////////////////////////////////////////////////////
      
  uniform bool in_cellFlag;      
  uniform vec3 in_cellSpacing;      
  uniform mat4 in_modelViewMatrix;      
  uniform mat4 in_projectionMatrix;      
  uniform mat4 in_volumeMatrix;      
      
  uniform vec3 in_volumeExtentsMin;      
  uniform vec3 in_volumeExtentsMax;      
      
  uniform mat4 in_inverseTextureDatasetMatrix;      
  uniform mat4 in_cellToPoint;      
  uniform vec3 in_textureExtentsMax;      
  uniform vec3 in_textureExtentsMin;      
      
  //This variable could be 'invariant varying' but it is declared      
  //as 'varying' to avoid compiler compatibility issues.      
  varying mat4 ip_inverseTextureDataAdjusted;

//VTK::Termination::Dec

//VTK::Cropping::Dec

//VTK::Shading::Dec

//////////////////////////////////////////////////////////////////////////////
///
/// Inputs
///
//////////////////////////////////////////////////////////////////////////////
attribute vec3 in_vertexPos;

//////////////////////////////////////////////////////////////////////////////
///
/// Outputs
///
//////////////////////////////////////////////////////////////////////////////
/// 3D texture coordinates for texture lookup in the fragment shader
varying vec3 ip_textureCoords;
varying vec3 ip_vertexPos;

void main()
{
  /// Get clipspace position
        
  vec4 pos = in_projectionMatrix * in_modelViewMatrix *      
             in_volumeMatrix * vec4(in_vertexPos.xyz, 1.0);      
  gl_Position = pos;

  /// Compute texture coordinates
  
  // For point dataset, we offset the texture coordinate       
  // to account for OpenGL treating voxel at the center of the cell.       
  vec3 uvx = sign(in_cellSpacing) * (in_vertexPos - in_volumeExtentsMin) /       
               (in_volumeExtentsMax - in_volumeExtentsMin);       
       
  if (in_cellFlag)       
    {       
    ip_textureCoords = uvx;       
    ip_inverseTextureDataAdjusted = in_inverseTextureDatasetMatrix;       
    }       
  else       
    {       
    // Transform cell tex-coordinates to point tex-coordinates       
    ip_textureCoords = (in_cellToPoint * vec4(uvx, 1.0)).xyz;       
    ip_inverseTextureDataAdjusted = in_cellToPoint * in_inverseTextureDatasetMatrix;       
    }

  /// Copy incoming vertex position for the fragment shader
  ip_vertexPos = in_vertexPos;
}
