////////////////////////////////////////////////////////////////////////////
//	File:		    SparseBundleCU.h
//	Author:		    Changchang Wu (ccwu@cs.washington.edu)
//	Description :   interface of the CUDA-version of multicore bundle adjustment
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

#if !defined(SPARSE_BUNDLE_CU_H)
#define SPARSE_BUNDLE_CU_H

#include "CuTexImage.h"
#include "ConfigBA.h"
#include "DataInterface.h"


class SparseBundleCU : public ParallelBA, public ConfigBA
{
protected:      //cpu data
    int             _num_camera;
    int             _num_point;
    int             _num_imgpt;
    CameraT*        _camera_data;   
    float*          _point_data;
    ////////////////////////////////
    const float*    _imgpt_data;
    const int*      _camera_idx;
    const int*      _point_idx;
	const int*		_focal_mask;
    vector<float>   _imgpt_datax;
    ////////////////////////
    float           _projection_sse;    //sumed square error
protected:      //cuda data
    CuTexImage_N      _cuCameraData;
    CuTexImage_N      _cuCameraDataEX;
    CuTexImage_N      _cuPointData;
    CuTexImage_N      _cuPointDataEX;
    CuTexImage_N      _cuMeasurements;
    CuTexImage_N      _cuImageProj;
    CuTexImage_N      _cuJacobianCamera;  
    CuTexImage_N      _cuJacobianPoint;  
    CuTexImage_N      _cuJacobianCameraT; 
    CuTexImage_N      _cuProjectionMap;
    CuTexImage_N      _cuPointMeasurementMap;  
    CuTexImage_N      _cuCameraMeasurementMap;  
    CuTexImage_N      _cuCameraMeasurementList; 
    CuTexImage_N      _cuCameraMeasurementListT; 

    ///////////////////////////////
    CuTexImage_N      _cuBufferData;
    ////////////////////////////
    CuTexImage_N      _cuBlockPC;
    CuTexImage_N      _cuVectorSJ;

    ///LM normal    equation
    CuTexImage_N      _cuVectorJtE;
    CuTexImage_N      _cuVectorJJ;
    CuTexImage_N      _cuVectorJX;
    CuTexImage_N      _cuVectorXK;
    CuTexImage_N      _cuVectorPK;
    CuTexImage_N      _cuVectorZK;
    CuTexImage_N      _cuVectorRK;

	///////////////////////
protected:
	int             _num_imgpt_q;
	float			_weight_q;
	CuTexImage_N		_cuCameraQList;
	CuTexImage_N		_cuCameraQMap;
	CuTexImage_N		_cuCameraQMapW;
	CuTexImage_N		_cuCameraQListW;
protected:
	bool		ProcessIndexCameraQ(vector<int>&qmap, vector<int>& qlist);
	void		ProcessWeightCameraQ(vector<int>&cpnum, vector<int>&qmap, vector<float>& qmapw, vector<float>&qlistw);

protected:      //internal functions
    int         GetParameterLength();
    int         InitializeBundle();
    int         ValidateInputData();
    void        ReleaseAllocatedData();
    bool        InitializeStorageForCG();
    bool        InitializeBundleGPU();
    bool        TransferDataToGPU();
    void        TransferDataToHost();
    void        DenormalizeData();
    void        NormalizeData();
    void        NormalizeDataF();
    void        NormalizeDataD();
    void        DebugProjections();
    void        RunDebugSteps();
    bool        CheckRequiredMem(int fresh = 1);
    bool        CheckRequiredMemX();
    void        ReserveStorage(size_t ncam, size_t npt, size_t nproj);
    void        ReserveStorageAuto();

protected:
    float       EvaluateProjection(CuTexImage_N& cam, CuTexImage_N&point, CuTexImage_N& proj);
    float       EvaluateProjectionX(CuTexImage_N& cam, CuTexImage_N&point, CuTexImage_N& proj);
    float       UpdateCameraPoint(CuTexImage_N& dx, CuTexImage_N& cuImageTempProj);
	float		SaveUpdatedSystem(float residual_reduction, float dx_sqnorm, float damping);
	float		EvaluateDeltaNorm();
    void        EvaluateJacobians(bool shuffle = true);
    void        PrepareJacobianNormalization();
    void        ComputeJtE(CuTexImage_N& E, CuTexImage_N& JtE, int mode = 0); 
    void        ComputeJX(CuTexImage_N& X, CuTexImage_N& JX, int mode = 0);
    void        ComputeDiagonal(CuTexImage_N& JJ, CuTexImage_N& JJI);
    void        ComputeBlockPC(float lambda, bool dampd = true);
    void        ApplyBlockPC(CuTexImage_N& v, CuTexImage_N& pv, int mode =0);
    int         SolveNormalEquationPCGB(float lambda);
    int         SolveNormalEquationPCGX(float lambda);
	int			SolveNormalEquation(float lambda);
	void		AdjustBundleAdjsutmentMode();
    void        NonlinearOptimizeLM();
    void        BundleAdjustment();
    void        RunTestIterationLM(bool reduced);
    void        SaveBundleRecord(int iter, float res, float damping, float& g_norm, float& g_inf);
    /////////////////////////////////
    void        SaveNormalEquation(float lambda);
    void        RunProfileSteps();
    void        WarmupDevice();
public:
    virtual float GetMeanSquaredError();
    virtual void SetCameraData(size_t ncam,  CameraT* cams);
    virtual void SetPointData(size_t npoint, Point3D* pts);
    virtual void SetProjection(size_t nproj, const Point2D* imgpts, const int* point_idx, const int* cam_idx);
	virtual void SetFocalMask(const int* fmask, float weight);
    virtual int  RunBundleAdjustment();

    ///
    virtual void AbortBundleAdjustment()                    {__abort_flag = true;}
    virtual int  GetCurrentIteration()                      {return __current_iteration; }
    virtual void SetNextTimeBudget(int seconds)             {__bundle_time_budget = seconds;}
	virtual void SetNextBundleMode(BundleModeT mode)		{__bundle_mode_next = mode;}
    virtual void SetFixedIntrinsics(bool fixed)             {__fixed_intrinsics = fixed; }
    virtual void EnableRadialDistortion(DistortionT type)   {__use_radial_distortion = type; }
    virtual void ParseParam(int narg, char** argv)          {ConfigBA::ParseParam(narg, argv); }
    virtual ConfigBA* GetInternalConfig()                   {return this; }
public:
    SparseBundleCU(int device);
	size_t  GetMemCapacity();
};

#endif

