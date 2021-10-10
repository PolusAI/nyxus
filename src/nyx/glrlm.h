#pragma once

class GLRLM
{
public:
	GLRLM() {}
	void initialize();

	// 1. Short Run Emphasis 
	double calc_SRE();
	// 2. Long Run Emphasis 
	double calc_LRE();
	// 3. Gray Level Non-Uniformity 
	double calc_GLN();
	// 4. Gray Level Non-Uniformity Normalized 
	double calc_GLNN();
	// 5. Run Length Non-Uniformity
	double calc_RLN();
	// 6. Run Length Non-Uniformity Normalized 
	double calc_RLNN();
	// 7. Run Percentage
	double calc_RP();
	// 8. Gray Level Variance 
	double calc_GLV();
	// 9. Run Variance 
	double calc_RV();
	// 10. Run Entropy 
	double calc_RE();
	// 11. Low Gray Level Run Emphasis 
	double calc_LGLRE();
	// 12. High Gray Level Run Emphasis 
	double calc_HGLRE();
	// 13. Short Run Low Gray Level Emphasis 
	double calc_SRLGLE();
	// 14. Short Run High Gray Level Emphasis 
	double calc_SRHGLE();
	// 15. Long Run Low Gray Level Emphasis 
	double calc_LRLGLE();
	// 16. Long Run High Gray Level Emphasis 
	double calc_LRHGLE();



protected:

};