#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <vector>
#include "hexagonality_and_polygonality.h"


// Returns  tuple {polyAve, hexAve, hexSd} :
//      polyAve - polygonality score
//      hexAve - hexagonality score
//      hexSd - dispersion of hexagonality score relative to its mean
//
std::tuple<double, double, double> Hexagonality_and_Polygonality_features::calculate (int num_neighbors, int roi_area, int roi_perimeter, double convhull_area, double min_feret_diam, double max_feret_diam)
{
    double polyAve = 0,    // wndchrm ratios[37]
        hexAve = 0,        // ratios[38]
        hexSd = 0;         // ratios[39]

    //--------------------------------Hexagonality/Polygonality-------------------
    //This section is a translation from the following Python code
    //https://github.com/LabShare/polus-plugins/blob/master/polus-feature-extraction-plugin/src/main.py

    int neighbors = num_neighbors; // NeighborUniqueIDs.size();
    double area = roi_area; // PixelsCount;
    double perimeter = roi_perimeter; // ROIPerimeter;
    double area_hull = convhull_area; // convexHullArea;
    double perim_hull = 6 * sqrt(area_hull / (1.5 * sqrt(3)));
    double perimeter_neighbors;

    if (neighbors == 0) 
        perimeter_neighbors = std::numeric_limits<double>::quiet_NaN();
    else 
        if (neighbors > 0) 
            perimeter_neighbors = perimeter / neighbors;

    //Polygonality metrics calculated based on the number of sides of the polygon
    if (neighbors > 2) 
    {

        double poly_size_ratio = 1.0 - abs(1.0 - perimeter_neighbors / sqrt((4 * area) / (neighbors / tan(M_PI / neighbors))));
        double poly_area_ratio = 1.0 - abs(1.0 - area / (0.25 * neighbors * perimeter_neighbors * perimeter_neighbors / tan(M_PI / neighbors)));

        //Calculate Polygonality Score
        double poly_ave = 10 * (poly_size_ratio + poly_area_ratio) / 2;

        //Hexagonality metrics calculated based on a convex, regular, hexagon
        double apoth1 = sqrt(3) * perimeter / 12;
        double apoth2 = sqrt(3) * max_feret_diam / 4;
        double apoth3 = min_feret_diam / 2;
        double side1 = perimeter / 6;
        double side2 = max_feret_diam / 2;
        double side3 = min_feret_diam / sqrt(3);
        double side4 = perim_hull / 6;

        //Unique area calculations from the derived and primary measures above
        double area1 = 0.5 * (3 * sqrt(3)) * side1 * side1;
        double area2 = 0.5 * (3 * sqrt(3)) * side2 * side2;
        double area3 = 0.5 * (3 * sqrt(3)) * side3 * side3;
        double area4 = 3 * side1 * apoth2;
        double area5 = 3 * side1 * apoth3;
        double area6 = 3 * side2 * apoth3;
        double area7 = 3 * side4 * apoth1;
        double area8 = 3 * side4 * apoth2;
        double area9 = 3 * side4 * apoth3;
        double area10 = area_hull;
        double area11 = area;

        //Create an array of all unique areas
        std::vector<double> list_area = { area1,area2,area3,area4,area5,area6,area7,area8,area9,area10,area11 };
        std::vector<double> area_array;

        //Create Summary statistics of all array ratios
        double sum = 0;
        for (int ib = 0; ib < list_area.size(); ++ib)
            for (int ic = ib + 1; ic < list_area.size(); ++ic) 
            {
                double area_ratio = 1.0 - abs(1.0 - list_area[ib] / list_area[ic]);
                area_array.push_back(area_ratio);
                sum += area_ratio;
            }
        double area_ratio_ave = sum / area_array.size();

        double sqrdTmp = 0;
        for (int i = 0; i < area_array.size(); ++i) 
        {
            sqrdTmp += (area_array[i] - area_ratio_ave) * (area_array[i] - area_ratio_ave);
        }
        double area_ratio_sd = sqrt(sqrdTmp / area_array.size());

        //Set the hexagon area ratio equal to the average Area Ratio
        double hex_area_ratio = area_ratio_ave;

        //Perimeter Ratio Calculations
        //Two extra apothems are now useful
        double apoth4 = sqrt(3) * perim_hull / 12;
        double apoth5 = sqrt(4 * area_hull / (4.5 * sqrt(3)));
        double perim1 = sqrt(24 * area / sqrt(3));
        double perim2 = sqrt(24 * area_hull / sqrt(3));
        double perim3 = perimeter;
        double perim4 = perim_hull;
        double perim5 = 3 * max_feret_diam;
        double perim6 = 6 * min_feret_diam / sqrt(3);
        double perim7 = 2 * area / apoth1;
        double perim8 = 2 * area / apoth2;
        double perim9 = 2 * area / apoth3;
        double perim10 = 2 * area / apoth4;
        double perim11 = 2 * area / apoth5;
        double perim12 = 2 * area_hull / apoth1;
        double perim13 = 2 * area_hull / apoth2;
        double perim14 = 2 * area_hull / apoth3;

        //Create an array of all unique Perimeters
        std::vector<double> list_perim = { perim1,perim2,perim3,perim4,perim5,perim6,perim7,perim8,perim9,perim10,perim11,perim12,perim13,perim14 };
        std::vector<double> perim_array;

        //Create an array of the ratio of all Perimeters to eachother
        //Create Summary statistics of all array ratios
        double sum2 = 0;
        for (int ib = 0; ib < list_perim.size(); ++ib)
            for (int ic = ib + 1; ic < list_perim.size(); ++ic) 
            {
                double perim_ratio = 1.0 - abs(1.0 - list_perim[ib] / list_perim[ic]);
                perim_array.push_back(perim_ratio);
                sum2 += perim_ratio;
            }
        double perim_ratio_ave = sum2 / perim_array.size();

        double sqrdTmp2 = 0;
        for (int i = 0; i < perim_array.size(); ++i) 
        {
            sqrdTmp2 += (perim_array[i] - perim_ratio_ave) * (perim_array[i] - perim_ratio_ave);
        }
        double perim_ratio_sd = sqrt(sqrdTmp2 / perim_array.size());

        //Set the HSR equal to the average Perimeter Ratio
        double hex_size_ratio = perim_ratio_ave;
        double hex_sd = sqrt((area_ratio_sd * area_ratio_sd + perim_ratio_sd * perim_ratio_sd) / 2);
        double hex_ave = 10 * (hex_area_ratio + hex_size_ratio) / 2;

        //ratios[37] = poly_ave;
        //ratios[38] = hex_ave;
        //ratios[39] = hex_sd;
        polyAve = poly_ave;
        hexAve = hex_ave;
        hexSd = hex_sd;
    }
    else 
        if (neighbors < 3) 
        {
            polyAve = std::numeric_limits<double>::quiet_NaN();
            hexAve = std::numeric_limits<double>::quiet_NaN();
            hexSd = std::numeric_limits<double>::quiet_NaN();
        }
    
    return { polyAve, hexAve, hexSd };
}

void Hexagonality_and_Polygonality_features::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        if (r.has_bad_data())
            continue;

        // Skip if the contour, convex hull, and neighbors are unavailable, otherwise the related features will be == NAN. Those feature will be equal to the default unassigned value.
        if (r.contour.contour_pixels.size() == 0 || r.convHull.CH.size() == 0 || r.fvals[NUM_NEIGHBORS][0] == 0)
            continue;

        Hexagonality_and_Polygonality_features hp;
        auto [polyAve, hexAve, hexSd] = hp.calculate (r.fvals[NUM_NEIGHBORS][0], r.raw_pixels.size(), r.fvals[PERIMETER][0], r.fvals[CONVEX_HULL_AREA][0], r.fvals[MIN_FERET_DIAMETER][0], r.fvals[MAX_FERET_DIAMETER][0]);
        r.fvals[POLYGONALITY_AVE][0] = polyAve;
        r.fvals[HEXAGONALITY_AVE][0] = hexAve;
        r.fvals[HEXAGONALITY_STDDEV][0] = hexSd;
    }
}
