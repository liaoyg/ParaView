/*=========================================================================

  Program:   ParaView
  Module:    vtkSMApplication.cxx

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkSMApplication.h"

#include "vtkClientServerStream.h"
#include "vtkDirectory.h"
#include "vtkKWArguments.h"
#include "vtkObjectFactory.h"
#include "vtkSMProxyManager.h"
#include "vtkSMXMLParser.h"
#include "vtkString.h"

#include "vtkProcessModule.h"

// Needed for VTK_USE_PATENTED
#include "vtkToolkits.h"
#include "vtkSMGeneratedModules.h"

#include "vtkSMDomain.h"
#include "vtkSMDomainIterator.h"
#include "vtkSMProperty.h"
#include "vtkSMPropertyIterator.h"
#include "vtkSMProxy.h"
#include "vtkSMProxyIterator.h"

vtkStandardNewMacro(vtkSMApplication);
vtkCxxRevisionMacro(vtkSMApplication, "1.9");

//---------------------------------------------------------------------------
vtkSMApplication::vtkSMApplication()
{
}

//---------------------------------------------------------------------------
vtkSMApplication::~vtkSMApplication()
{
}

extern "C" { void vtkPVServerManager_Initialize(vtkClientServerInterpreter*); }

//---------------------------------------------------------------------------
void vtkSMApplication::Initialize()
{
//   args->AddCallback("--configuration-path", 
//                     vtkKWArguments::EQUAL_ARGUMENT, 
//                     NULL, 
//                     NULL, 
//                     "Directory where all configuration files are stored");

//   args->Parse();

  vtkPVServerManager_Initialize(
    vtkProcessModule::GetProcessModule()->GetInterpreter());

  vtkSMProxyManager* proxyM = vtkSMProxyManager::New();
  this->SetProxyManager(proxyM);

  vtkSMXMLParser* parser = vtkSMXMLParser::New();

  char* init_string;

  init_string =  vtkSMDefaultModulesfiltersGetInterfaces();
  parser->Parse(init_string);
  parser->ProcessConfiguration(proxyM);
  delete[] init_string;

  init_string =  vtkSMDefaultModulesreadersGetInterfaces();
  parser->Parse(init_string);
  parser->ProcessConfiguration(proxyM);
  delete[] init_string;

  init_string =  vtkSMDefaultModulessourcesGetInterfaces();
  parser->Parse(init_string);
  parser->ProcessConfiguration(proxyM);
  delete[] init_string;

  init_string =  vtkSMDefaultModulesutilitiesGetInterfaces();
  parser->Parse(init_string);
  parser->ProcessConfiguration(proxyM);
  delete[] init_string;

  init_string =  vtkSMDefaultModulesrenderingGetInterfaces();
  parser->Parse(init_string);
  parser->ProcessConfiguration(proxyM);
  delete[] init_string;

  parser->Delete();

  proxyM->InstantiateGroupPrototypes("filters");
  proxyM->InstantiateGroupPrototypes("readers");
  proxyM->InstantiateGroupPrototypes("sources");
  proxyM->InstantiateGroupPrototypes("utilities");
  proxyM->InstantiateGroupPrototypes("rendering");

  cout << "Check for properties without domains" << endl;
  vtkSMProxyIterator* pi = vtkSMProxyIterator::New();
  vtkSMPropertyIterator* pri = vtkSMPropertyIterator::New();
  vtkSMDomainIterator* di = vtkSMDomainIterator::New();
  for ( pi->Begin(); !pi->IsAtEnd(); pi->Next() )
    {
    vtkSMProxy* proxy = pi->GetProxy();
    pri->SetProxy(proxy);
    for ( pri->Begin(); !pri->IsAtEnd(); pri->Next() )
      {
      vtkSMProperty* prop = pri->GetProperty();
      di->SetProperty(prop);
      int count = 0;
      for ( di->Begin(); !di->IsAtEnd(); di->Next() )
        {
        count ++;
        }
      if ( !count )
        {
        cout << "* Property: " << proxy->GetXMLName() << "->" << prop->GetXMLName() << " has no domain" << endl;
        }
      }
    }
  pi->Delete();
  pri->Delete();
  di->Delete();
  cout << "Done checking for properties without domains" << endl;

// //  const char* directory = args->GetValue("--configuration-path");
//   const char* directory =  "/home/berk/etc/servermanager";
//   if (directory)
//     {
//     vtkDirectory* dir = vtkDirectory::New();
//     if(!dir->Open(directory))
//       {
//       dir->Delete();
//       return;
//       }
    
//     for(int i=0; i < dir->GetNumberOfFiles(); ++i)
//       {
//       const char* file = dir->GetFile(i);
//       int extPos = vtkString::Length(file)-4;
      
//       // Look for the ".xml" extension.
//       if((extPos > 0) && vtkString::Equals(file+extPos, ".xml"))
//         {
//         char* fullPath 
//           = new char[vtkString::Length(file)+vtkString::Length(directory)+2];
//         strcpy(fullPath, directory);
//         strcat(fullPath, "/");
//         strcat(fullPath, file);
        
//         cerr << fullPath << endl;
        
//         vtkSMXMLParser* parser = vtkSMXMLParser::New();
//         parser->SetFileName(fullPath);
//         parser->Parse();
//         parser->ProcessConfiguration(proxyM);
//         parser->Delete();
        
//         delete [] fullPath;
//         }
//       }
//     dir->Delete();
//     }
  
  proxyM->Delete();
}

//---------------------------------------------------------------------------
int vtkSMApplication::ParseConfigurationFile(const char* fname, const char* dir)
{
  vtkSMProxyManager* proxyM = this->GetProxyManager();
  if (!proxyM)
    {
    vtkErrorMacro("No global proxy manager defined. Can not parse file");
    return 0;
    }

  ostrstream tmppath;
  tmppath << dir << "/" << fname << ends;
  vtkSMXMLParser* parser = vtkSMXMLParser::New();
  parser->SetFileName(tmppath.str());
  delete[] tmppath.str();
  int res = parser->Parse();
  parser->ProcessConfiguration(proxyM);
  parser->Delete();
  return res;
}

//---------------------------------------------------------------------------
void vtkSMApplication::Finalize()
{
  //this->GetProcessModule()->FinalizeInterpreter();
  this->SetProxyManager(0);

}

//---------------------------------------------------------------------------
void vtkSMApplication::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
