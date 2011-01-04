/*=========================================================================

  Program:   ParaView
  Module:    vtkSMRemoteObject.cxx

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkSMRemoteObject.h"

#include "vtkObjectFactory.h"
#include "vtkProcessModule.h"
#include "vtkSMMessage.h"
#include "vtkSMSession.h"

#include <vtksys/SystemTools.hxx>
#include <vtksys/ios/sstream>

//----------------------------------------------------------------------------
vtkSMRemoteObject::vtkSMRemoteObject()
{
  this->GlobalID = 0;
  this->GlobalIDString = NULL;
  this->Location = 0;
  this->Session = NULL;
}

//----------------------------------------------------------------------------
vtkSMRemoteObject::~vtkSMRemoteObject()
{
  if(this->Session && this->GlobalID != 0)
    {
    this->Session->UnRegisterRemoteObject(this);
    }
  this->SetSession(0);
  delete [] this->GlobalIDString;
  this->GlobalIDString = NULL;
}

//----------------------------------------------------------------------------
vtkSMSession* vtkSMRemoteObject::GetSession()
{
  return this->Session;
}

//----------------------------------------------------------------------------
void vtkSMRemoteObject::SetSession(vtkSMSession* session)
{
  if (this->Session != session)
    {
    this->Session = session;
    this->Modified();
    }
  // Register object if possible
  if(this->Session && this->GlobalID != 0)
    {
    this->Session->RegisterRemoteObject(this);
    }
}

//----------------------------------------------------------------------------
vtkSMProxyManager* vtkSMRemoteObject::GetProxyManager()
{
  return (this->Session? this->Session->GetProxyManager() : NULL);
}

//----------------------------------------------------------------------------
void vtkSMRemoteObject::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Session: " << this->Session << endl;
  os << indent << "GlobalID: " << this->GlobalID << endl;
}

//---------------------------------------------------------------------------
vtkTypeUInt32 vtkSMRemoteObject::GetGlobalID()
{
  if (this->Session != NULL && this->GlobalID == 0)
    {
    this->GlobalID = this->GetSession()->GetNextGlobalUniqueIdentifier();
    // Register object
    this->Session->RegisterRemoteObject(this);

    vtksys_ios::ostringstream cname;
    cname << this->GlobalID;

    delete [] this->GlobalIDString;
    this->GlobalIDString = vtksys::SystemTools::DuplicateString(
      cname.str().c_str());
    }

  return this->GlobalID;
}

//---------------------------------------------------------------------------
const char* vtkSMRemoteObject::GetGlobalIDAsString()
{
  if (!this->GlobalIDString)
    {
    this->GetGlobalID();
    }

  return this->GlobalIDString;
}

//---------------------------------------------------------------------------
void vtkSMRemoteObject::SetGlobalID(vtkTypeUInt32 guid)
{
  // Unregister current object with previous ID if already registered
  if(this->GlobalID != 0 && this->Session &&
     this->Session->GetRemoteObject(this->GlobalID) == this)
    {
    this->Session->UnRegisterRemoteObject(this);
    }

  // Keep new ID
  this->GlobalID = guid;

  // Register object if possible
  if(this->Session && this->GlobalID != 0)
    {
    this->Session->RegisterRemoteObject(this);
    }
}

//---------------------------------------------------------------------------
void vtkSMRemoteObject::PushState(vtkSMMessage* msg)
{
  if(this->Location == 0)
    {
    return; // This object is a prototype and has no location
    }

  // Check if a GUID has been assigned to that object otherwise assign a new one
  vtkTypeUInt32 gid = this->GetGlobalID();
  msg->set_global_id(gid);
  msg->set_location(this->Location);
  if (this->GetSession())
    {
    this->GetSession()->PushState(msg);
    }
  else // FIXME this case occurs for prototype...
    {
    vtkErrorMacro("No session found");
    // FIXME Throw exception or error feed back : Not PVSession found !
    }
}

//---------------------------------------------------------------------------
bool vtkSMRemoteObject::PullState(vtkSMMessage* msg)
{
  if(this->Location == 0)
    {
    return true; // This object is a prototype and has no location
    }

  msg->set_global_id(this->GlobalID);
  msg->set_location(this->Location);
  if(this->GetSession())
    {
    this->GetSession()->PullState(msg);
    }
  else
    {
    vtkErrorMacro("No session found");
    // FIXME Throw exception or error feed back : Not PVSession found !
    return false;
    }
  return true; // Successful call
}

//---------------------------------------------------------------------------
void vtkSMRemoteObject::Invoke(vtkSMMessage* msg)
{
  msg->set_global_id(this->GlobalID);
  msg->set_location(this->Location);
  if(this->GetSession())
    {
    this->GetSession()->Invoke(msg);
    }
  else
    {
    vtkErrorMacro("No session found");
    }
}

//---------------------------------------------------------------------------
void vtkSMRemoteObject::DestroyPMObject()
{
  Message msg;
  msg.set_global_id(this->GlobalID);
  msg.set_location(this->Location);
  this->GetSession()->DeletePMObject(&msg);
}
