namespace NyxusGpu
{
	// CPP-side-facing interface to CUDA feature calculator
	//		(per-ROI)
	bool GLSZMfeature_calc (
		// out
		double& fv_SAE,
		double& fv_LAE,
		double& fv_GLN,
		double& fv_GLNN,
		double& fv_SZN,
		double& fv_SZNN,
		double& fv_ZP,
		double& fv_GLV,
		double& fv_ZV,
		double& fv_ZE,
		double& fv_LGLZE,
		double& fv_HGLZE,
		double& fv_SALGLE,
		double& fv_SAHGLE,
		double& fv_LALGLE,
		double& fv_LAHGLE,
		// in
		int Ng, 
		int Ns, 
		PixIntens* I, 
		int* P, 
		double sum_p, 
		size_t nnz, 
		double EPS);
}