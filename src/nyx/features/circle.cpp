#include <algorithm>
#include "circle.h"

EnclosingInscribingCircumscribingCircleFeature::EnclosingInscribingCircumscribingCircleFeature() : FeatureMethod("EnclosingInscribingCircumscribingCircleFeature")
{
    provide_features({ DIAMETER_MIN_ENCLOSING_CIRCLE, DIAMETER_INSCRIBING_CIRCLE, DIAMETER_CIRCUMSCRIBING_CIRCLE });
    add_dependencies({ PERIMETER, CENTROID_X, CENTROID_Y });    // Availability of feature 'PERIMETER' ensures availability of LR::contour
}

void EnclosingInscribingCircumscribingCircleFeature::calculate(LR& r)
{
    d_minEnclo = calculate_min_enclosing_circle_diam (r.contour);
    std::tie(d_inscr, d_circum) = calculate_inscribing_circumscribing_circle (r.contour, r.fvals[CENTROID_X][0], r.fvals[CENTROID_Y][0]);
}

void EnclosingInscribingCircumscribingCircleFeature::save_value(std::vector<std::vector<double>>& fvals)
{
    fvals[DIAMETER_MIN_ENCLOSING_CIRCLE][0] = d_minEnclo;
    fvals[DIAMETER_INSCRIBING_CIRCLE][0] = d_inscr;
    fvals[DIAMETER_CIRCUMSCRIBING_CIRCLE][0] = d_circum;
}

double EnclosingInscribingCircumscribingCircleFeature::calculate_min_enclosing_circle_diam (std::vector<Pixel2>& Contour)
{
    // Inspired by https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L166
    
    // Find the minimum enclosing circle of an object
    Point2f center;
    float radius = 0;
    minEnclosingCircle(Contour, center, radius);

    // Diameter of the minimum circumference of the projection area.
    double diameter_min_enclosing_circle = 2 * radius;
    return diameter_min_enclosing_circle;
}

void EnclosingInscribingCircumscribingCircleFeature::findCircle3pts (const std::vector<Pixel2>& pts, Point2f& center, float& radius)
{
    // 2 edges of the triangle v1, v2
    Point2f v1 = pts[1] - pts[0],
        v2 = pts[2] - pts[0];

    // Center is intersection of midperpendicular lines of the two edges v1, v2
    //      a1*x + b1*y = c1 where a1 = v1.x, b1 = v1.y
    //      a2*x + b2*y = c2 where a2 = v2.x, b2 = v2.y

    Point2f midPoint1 = (pts[0] + pts[1]) / 2.0f;
    float c1 = midPoint1.x * v1.x + midPoint1.y * v1.y;
    Point2f midPoint2 = (pts[0] + pts[2]) / 2.0f;
    float c2 = midPoint2.x * v2.x + midPoint2.y * v2.y;
    float det = v1.x * v2.y - v1.y * v2.x;
    if (fabs(det) <= EPS)
    {
        // v1 and v2 are colinear, so the longest distance between any 2 points is the diameter of the minimum enclosing circle
        float d1 = normL2 (pts[0] - pts[1]),    // Note: d1-3 are squared distances
            d2 = normL2 (pts[0] - pts[2]),
            d3 = normL2 (pts[1] - pts[2]);
        radius = sqrt(std::max(d1, std::max(d2, d3))) * 0.5f + EPS;
        if (d1 >= d2 && d1 >= d3)
        {
            center = (pts[0] + pts[1]) * 0.5f;
        }
        else if (d2 >= d1 && d2 >= d3)
        {
            center = (pts[0] + pts[2]) * 0.5f;
        }
        else
        {
            center = (pts[1] + pts[2]) * 0.5f;
        }
        return;
    }
    float cx = (c1 * v2.y - c2 * v1.y) / det;
    float cy = (v1.x * c2 - v2.x * c1) / det;
    center.x = (float)cx;
    center.y = (float)cy;
    cx -= pts[0].x;
    cy -= pts[0].y;
    radius = (float)(std::sqrt(cx * cx + cy * cy)) + EPS;
}

void EnclosingInscribingCircumscribingCircleFeature::findThirdPoint (const std::vector<Pixel2>& pts, int i, int j, Point2f& center, float& radius)
{
    center.x = (float)(pts[j].x + pts[i].x) / 2.0f;
    center.y = (float)(pts[j].y + pts[i].y) / 2.0f;
    float dx = (float)(pts[j].x - pts[i].x);
    float dy = (float)(pts[j].y - pts[i].y);
    radius = (float)normL2(Point2f(dx, dy)) / 2.0f + EPS;

    for (int k = 0; k < j; k++)
    {
        dx = center.x - (float)pts[k].x;
        dy = center.y - (float)pts[k].y;
        if (normL2(Point2f(dx, dy)) < radius)
            continue;
        else
        {
            std::vector<Pixel2> ptsf;   
            ptsf.push_back (Pixel2(pts[i].x, pts[i].y, 0));   
            ptsf.push_back(Pixel2(pts[j].x, pts[j].y, 0));   
            ptsf.push_back(Pixel2(pts[k].x, pts[k].y, 0));  
            Point2f new_center; float new_radius = 0;
            findCircle3pts(ptsf, new_center, new_radius);
            if (new_radius > 0)
            {
                radius = new_radius;
                center = new_center;
            }
        }
    }
}

void EnclosingInscribingCircumscribingCircleFeature::findSecondPoint (const std::vector<Pixel2>& pts, int i, Point2f& center, float& radius)
{
    center.x = (float)(pts[0].x + pts[i].x) / 2.0f;
    center.y = (float)(pts[0].y + pts[i].y) / 2.0f;
    float dx = (float)(pts[0].x - pts[i].x);
    float dy = (float)(pts[0].y - pts[i].y);
    radius = (float)normL2(Point2f(dx, dy)) / 2.0f + EPS;

    for (int j = 1; j < i; j++)
    {
        dx = center.x - (float)pts[j].x;
        dy = center.y - (float)pts[j].y;
        if (normL2(Point2f(dx, dy)) < radius)
            continue;
        else
        {
            Point2f new_center; float new_radius = 0;
            findThirdPoint(pts, i, j, new_center, new_radius);
            if (new_radius > 0)
            {
                radius = new_radius;
                center = new_center;
            }
        }
    }
}

void EnclosingInscribingCircumscribingCircleFeature::findMinEnclosingCircle (std::vector<Pixel2>& contour, Point2f& center, float& radius)
{
    center.x = (float)(contour[0].x + contour[1].x) / 2.0f;
    center.y = (float)(contour[0].y + contour[1].y) / 2.0f;
    float dx = (float)(contour[0].x - contour[1].x),
        dy = (float)(contour[0].y - contour[1].y);
    radius = (float)normL2(Point2f(dx, dy)) / 2.0f + EPS;

    auto count = contour.size();
    for (auto i = 2; i < count; i++)
    {
        dx = (float)contour[i].x - center.x;
        dy = (float)contour[i].y - center.y;
        float d = (float)normL2(Point2f(dx, dy));
        if (d < radius)
            continue;
        else
        {
            Point2f new_center; 
            float new_radius = 0;
            findSecondPoint(contour, i, new_center, new_radius);
            if (new_radius > 0)
            {
                radius = new_radius;
                center = new_center;
            }
        }
    }
}

// See Welzl, Emo. Smallest enclosing disks (balls and ellipsoids). Springer Berlin Heidelberg, 1991.
void EnclosingInscribingCircumscribingCircleFeature::minEnclosingCircle(
	// in:
	std::vector<Pixel2>& Contour,
	// out:
	Point2f & _center, float& _radius)
{
    _center.x = _center.y = 0.f;
    _radius = 0.f;
    auto count = Contour.size();

    if (count == 0)
        return;

    switch (count)
    {
        case 1:
        {
            _center = Point2f ((float)Contour[0].x, (float)Contour[0].y);   //(is_float) ? ptsf[0] : Point2f((float)ptsi[0].x, (float)ptsi[0].y);
            _radius = EPS;
            break;
        }
        case 2:
        {
            Point2f p1 = Point2f ((float)Contour[0].x, (float)Contour[0].y);   //(is_float) ? ptsf[0] : Point2f((float)ptsi[0].x, (float)ptsi[0].y);
            Point2f p2 = Point2f ((float)Contour[1].x, (float)Contour[1].y);    //(is_float) ? ptsf[1] : Point2f((float)ptsi[1].x, (float)ptsi[1].y);
            _center.x = (p1.x + p2.x) / 2.0f;
            _center.y = (p1.y + p2.y) / 2.0f;
            _radius = (float)(normL2(p1 - p2) / 2.0) + EPS;
            break;
        }
        default:
        {
            Point2f center;
            float radius = 0.f;

            findMinEnclosingCircle (Contour, center, radius);
            _center = center;
            _radius = radius;
            break;
        }
    }
}


std::tuple <double, double> EnclosingInscribingCircumscribingCircleFeature::calculate_inscribing_circumscribing_circle (std::vector<Pixel2> & contours, double xCentroid, double yCentroid)
{
    //-----------------circumscribing and inscribing circle ---------------------------
    //https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L199

    double yCentroid2 = yCentroid - 1;
    double xCentroid2 = xCentroid - 1;
    std::vector <double> distances;

    for (size_t j = 0; j < contours.size(); j++) 
    {
        double tmpx = (contours[j].x - xCentroid2);
        double tmpy = (contours[j].y - yCentroid2);
        double distance = sqrt(tmpx * tmpx + tmpy * tmpy);
        distances.push_back(distance);
    }

    double radius_circumscribing_circle = *std::max_element(distances.begin(), distances.end());
    double radius_inscribing_circle = *std::min_element(distances.begin(), distances.end());

    double diameter_circumscribing_circle = 2 * radius_circumscribing_circle;
    double diameter_inscribing_circle = 2 * radius_inscribing_circle;

    return { diameter_inscribing_circle, diameter_circumscribing_circle };
}

void EnclosingInscribingCircumscribingCircleFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        if (r.has_bad_data())
            continue;

        // Skip if the contour, convex hull, and neighbors are unavailable, otherwise the related features will be == NAN. Those feature will be equal to the default unassigned value.
        if (r.contour.size() == 0)
            continue;

        EnclosingInscribingCircumscribingCircleFeature cir;
        cir.calculate(r);
        cir.save_value(r.fvals);
    }
}

// Not using the online mode for this feature
void EnclosingInscribingCircumscribingCircleFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not supporting the online mode for this class

void EnclosingInscribingCircumscribingCircleFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
    calculate(r);
}