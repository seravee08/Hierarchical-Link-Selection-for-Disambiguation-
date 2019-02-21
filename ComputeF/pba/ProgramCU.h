////////////////////////////////////////////////////////////////////////////
//  File:           ProgramCU.h
//  Author:         Changchang Wu
//  Description :   interface for the ProgramCU classes.
//                  It is basically a wrapper around all the CUDA kernels
//
//  Copyright (c) 2011  Changchang Wu (ccwu@cs.washington.edu)
//    and the University of Washington at Seattle 
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation; either
//  Version 3 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _PROGRAM_CU_H
#define _PROGRAM_CU_H

class CuTexImage_N;

namespace ProgramCU
{
    /////////////////////////////////////////////////////////
    int    SetCudaDevice(int device);
    size_t GetCudaMemoryCap();
    int    CheckErrorCUDA(const char* location);
    void   FinishWorkCUDA();
    void   ClearPreviousError();
	void   ResetCurrentDevice();
    void   GetBlockConfiguration(unsigned int nblock, unsigned int& bw, unsigned int& bh);

    //////////////////////////////////////////////////////////
    void   ComputeSQRT(CuTexImage_N& tex);
    void   ComputeRSQRT(CuTexImage_N& tex);
    void   ComputeVXY(CuTexImage_N& texX, CuTexImage_N& texY, CuTexImage_N& result, unsigned int part =0, unsigned int skip = 0);
    void   ComputeSAXPY(float a, CuTexImage_N& texX, CuTexImage_N& texY, CuTexImage_N& result); 
    void   ComputeSAX(float a, CuTexImage_N& texX, CuTexImage_N& result); 
    void   ComputeSXYPZ(float a, CuTexImage_N& texX, CuTexImage_N& texY, CuTexImage_N& texZ, CuTexImage_N& result);
    float  ComputeVectorMax(CuTexImage_N& vector, CuTexImage_N& buf); 
    float  ComputeVectorSum(CuTexImage_N& vector, CuTexImage_N& buf, int skip);
    double ComputeVectorNorm(CuTexImage_N& vector, CuTexImage_N& buf); 
    double ComputeVectorNormW(CuTexImage_N& vector, CuTexImage_N& weight, CuTexImage_N& buf); 
    double ComputeVectorDot(CuTexImage_N& vector1, CuTexImage_N& vector2, CuTexImage_N& buf); 

    //////////////////////////////////////////////////////////////////////////
    void  UncompressCamera(int ncam, CuTexImage_N& camera0, CuTexImage_N& result);
    void  CompressCamera(int ncam, CuTexImage_N& camera0, CuTexImage_N& result);
    void  UpdateCameraPoint(int ncam, CuTexImage_N& camera, CuTexImage_N& point,
                            CuTexImage_N& delta, CuTexImage_N& new_camera, CuTexImage_N& new_point, int mode = 0);

    /////////////////////////////////////////////////////////////////////////
    void  ComputeJacobian(CuTexImage_N& camera, CuTexImage_N& point, CuTexImage_N& jc, 
                          CuTexImage_N& jp, CuTexImage_N& proj_map, CuTexImage_N& sj,
                          CuTexImage_N& meas, CuTexImage_N& cmlist,
                          bool intrinsic_fixed, int radial_distortion, bool shuffle);
    void  ComputeProjection(CuTexImage_N& camera, CuTexImage_N& point, CuTexImage_N& meas, 
                            CuTexImage_N& proj_map, CuTexImage_N& proj, int radial);
    void  ComputeProjectionX(CuTexImage_N& camera, CuTexImage_N& point, CuTexImage_N& meas, 
                            CuTexImage_N& proj_map, CuTexImage_N& proj, int radial);

    bool  ShuffleCameraJacobian(CuTexImage_N& jc, CuTexImage_N& map, CuTexImage_N& result); 

    /////////////////////////////////////////////////////////////
    void  ComputeDiagonal(CuTexImage_N& jc, CuTexImage_N& cmap, CuTexImage_N& jp, CuTexImage_N& pmap, 
          CuTexImage_N& cmlist, CuTexImage_N& jtjd, CuTexImage_N& jtjdi, 
		  bool jc_transpose, int radial, bool add_existing_diagc); 
    void  MultiplyBlockConditioner(int ncam, int npoint, CuTexImage_N& blocks, 
          CuTexImage_N& vector, CuTexImage_N& result, int radial, int mode = 0);

	////////////////////////////////////////////////////////////////////////////////
    void  ComputeProjectionQ(CuTexImage_N& camera, CuTexImage_N& qmap, CuTexImage_N& qw, CuTexImage_N& proj, int offset);
	void  ComputeJQX(CuTexImage_N& x, CuTexImage_N& qmap,  CuTexImage_N& wq,CuTexImage_N& sj, CuTexImage_N& jx, int offset);
    void  ComputeJQtEC(CuTexImage_N& pe, CuTexImage_N& qlist, CuTexImage_N& wq, CuTexImage_N& sj, CuTexImage_N& result);
	void  ComputeDiagonalQ(CuTexImage_N& qlistw, CuTexImage_N&sj, CuTexImage_N& diag);

    //////////////////////////////////////////////////////////////////////////
    void  ComputeJX(int point_offset, CuTexImage_N& x, CuTexImage_N& jc, CuTexImage_N& jp, 
                                      CuTexImage_N& jmap, CuTexImage_N& result, int mode = 0);
    void  ComputeJtE(CuTexImage_N& pe, CuTexImage_N& jc, CuTexImage_N& cmap, CuTexImage_N& cmlist,
          CuTexImage_N& jp, CuTexImage_N& pmap, CuTexImage_N& jte, bool jc_transpose, int mode = 0); 
    void  ComputeDiagonalBlock(float lambda, bool dampd, CuTexImage_N& jc, CuTexImage_N& cmap,
          CuTexImage_N& jp, CuTexImage_N& pmap, CuTexImage_N& cmlist, CuTexImage_N& diag, CuTexImage_N& blocks,
          int radial_distortion, bool jc_transpose, bool add_existing_diagc, int mode = 0); 

    /////////////////////////////////////////////////////////////////////
    void  ComputeJX_(CuTexImage_N& x,  CuTexImage_N& jx, CuTexImage_N& camera, CuTexImage_N& point, CuTexImage_N& meas,
                     CuTexImage_N& pjmap, bool intrinsic_fixed, int radial_distortion, int mode = 0);
    void  ComputeJtE_(CuTexImage_N& e,  CuTexImage_N& jte, CuTexImage_N& camera, CuTexImage_N& point,
                      CuTexImage_N& meas, CuTexImage_N& cmap,CuTexImage_N& cmlist,  CuTexImage_N& pmap, 
                      CuTexImage_N& jmap, CuTexImage_N& jp, bool intrinsic_fixed, int radial_distortion, int mode = 0);
    void  ComputeDiagonalBlock_(float lambda, bool dampd, CuTexImage_N& camera, CuTexImage_N& point, CuTexImage_N& meas, 
                                CuTexImage_N& cmap,CuTexImage_N& cmlist,  CuTexImage_N& pmap, CuTexImage_N& jmap, CuTexImage_N& jp, 
                                CuTexImage_N& sj, CuTexImage_N& diag, CuTexImage_N& blocks, 
								bool intrinsic_fixed, int radial_distortion, 
								bool add_existing_diagc, int mode = 0);

};

#endif

