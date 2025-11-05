#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <vector>
#include "hexagonality_polygonality.h"

using namespace Nyxus;

HexagonalityPolygonalityFeature::HexagonalityPolygonalityFeature() : FeatureMethod("HexagonalityPolygonalityFeature")
{
    provide_features({ Feature2D::POLYGONALITY_AVE, Feature2D::HEXAGONALITY_AVE, Feature2D::HEXAGONALITY_STDDEV });
    add_dependencies({ Feature2D::NUM_NEIGHBORS, Feature2D::PERIMETER, Feature2D::CONVEX_HULL_AREA, Feature2D::STAT_FERET_DIAM_MAX, Feature2D::STAT_FERET_DIAM_MIN });
}

void HexagonalityPolygonalityFeature::calculate (LR& r, const Fsettings& s)
{
    // The whole calculation is inspired by calculation of this feature in POLUS feature extraction plugin 
    // https://github.com/PolusAI/image-tools/tree/master/features/polus-feature-extraction-plugin
    //
    //  Interpreting feature values:
    // 
    //  Polygonality score: 
    //      The score ranges from -infinity to 10. Score 10 indicates 
    //      the object shape is a polygon and score -infinity indicates the object shape 
    //      is not polygon.
    //      
     // Hexagonality score: 
    //      The score ranges from -infinity to 10. Score 10 indicates the object shape 
    //      is hexagon and score - infinity indicates the object shape is not hexagon.
    //
    //  Hexagonality standard deviation:
    //      Dispersion of hexagonality_score relative to its mean.
    //

    size_t neighbors = r.fvals[(int)Feature2D::NUM_NEIGHBORS][0];
    double area = r.aux_area; 
    double perimeter = r.fvals[(int)Feature2D::PERIMETER][0];
    double area_hull = r.fvals[(int)Feature2D::CONVEX_HULL_AREA][0];
    double perim_hull = 6 * sqrt(area_hull / (1.5 * sqrt(3)));
    double min_feret_diam = r.fvals[(int)Feature2D::STAT_FERET_DIAM_MIN][0];
    double max_feret_diam = r.fvals[(int)Feature2D::STAT_FERET_DIAM_MAX][0];
    double perimeter_neighbors;

    if (neighbors == 0)
        perimeter_neighbors = HexagonalityPolygonalityFeature::novalue;
    else
        perimeter_neighbors = perimeter / neighbors;

    // Polygonality metrics calculated based on the number of sides of the polygon
    if (neighbors > 2)
    {
        double poly_size_ratio = 1.0 - abs(1.0 - perimeter_neighbors / sqrt((4 * area) / (neighbors / tan(M_PI / neighbors))));
        double poly_area_ratio = 1.0 - abs(1.0 - area / (0.25 * neighbors * perimeter_neighbors * perimeter_neighbors / tan(M_PI / neighbors)));

        // Calculate Polygonality Score
        double poly_ave = 10 * (poly_size_ratio + poly_area_ratio) / 2;

        // Hexagonality metrics calculated based on a convex, regular, hexagon
        double apoth1 = sqrt(3) * perimeter / 12;
        double apoth2 = sqrt(3) * max_feret_diam / 4;
        double apoth3 = min_feret_diam / 2;
        double side1 = perimeter / 6;
        double side2 = max_feret_diam / 2;
        double side3 = min_feret_diam / sqrt(3);
        double side4 = perim_hull / 6;

        // Unique area calculations from the derived and primary measures above
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

        // Create an array of all unique areas
        std::vector<double> list_area = { area1, area2, area3, area4, area5, area6, area7, area8, area9, area10, area11 };
        std::vector<double> area_array;

        // Create Summary statistics of all array ratios
        double sum = 0;
        for (int ib = 0; ib < list_area.size(); ++ib)
            for (int ic = ib + 1; ic < list_area.size(); ++ic)
            {
                double area_ratio = 1.0 - abs(1.0 - list_area[ib] / list_area[ic]);
                
                // skip NANs
                if (!std::isfinite(area_ratio))
                    continue;

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

        // Set the hexagon area ratio equal to the average Area Ratio
        double hex_area_ratio = area_ratio_ave;

        // Perimeter Ratio Calculations. Two extra apothems are now useful
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

        // Create an array of all unique Perimeters
        std::vector<double> list_perim = { perim1, perim2, perim3, perim4, perim5, perim6, perim7, perim8, perim9, perim10, perim11, perim12, perim13, perim14 };
        std::vector<double> perim_array; 

        // 1 - Create an array of the ratio of all Perimeters to each other. 2 - Create Summary statistics of all array ratios.
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

        polyAve = poly_ave;
        hexAve = hex_ave;
        hexSd = hex_sd;
    }
    else
        if (neighbors < 3)
        {
            polyAve = HexagonalityPolygonalityFeature::novalue;
            hexAve = HexagonalityPolygonalityFeature::novalue;
            hexSd = HexagonalityPolygonalityFeature::novalue;
        }
}

void HexagonalityPolygonalityFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}

void HexagonalityPolygonalityFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader&)
{
    // This feature doesn't contain ROI size-critical sections, so we're using the trivial ROI's calculate()
    calculate (r, s);
}

void HexagonalityPolygonalityFeature::save_value(std::vector<std::vector<double>>& fvals)
{
    fvals[(int)Feature2D::POLYGONALITY_AVE][0] = polyAve;
    fvals[(int)Feature2D::HEXAGONALITY_AVE][0] = hexAve;
    fvals[(int)Feature2D::HEXAGONALITY_STDDEV][0] = hexSd;
}

void HexagonalityPolygonalityFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        // Feasibility check #1
        if (r.has_bad_data())
        {
            // Explicitly assign dummy yet valid values to indicate that features weren't calculated. Better than NAN - less data cleaning before training
            HexagonalityPolygonalityFeature hexpo;
            hexpo.polyAve = hexpo.hexAve = hexpo.hexSd = HexagonalityPolygonalityFeature::novalue;
            hexpo.save_value(r.fvals);

            // Skip feature calculation
            continue;
        }

        // Feasibility check #2
        if (r.contour.size() == 0 || r.fvals[(int)Feature2D::CONVEX_HULL_AREA][0] == 0 || r.fvals[(int)Feature2D::NUM_NEIGHBORS][0] == 0)
        {
            // Explicitly assign dummy yet valid values to indicate that features weren't calculated. Better than NAN - less data cleaning before training
            HexagonalityPolygonalityFeature hexpo;
            hexpo.polyAve = hexpo.hexAve = hexpo.hexSd = HexagonalityPolygonalityFeature::novalue;
            hexpo.save_value(r.fvals);

            // Skip feature calculation
            continue;
        }

        HexagonalityPolygonalityFeature f;
        f.calculate (r, s);
        f.save_value (r.fvals);
    }
}




