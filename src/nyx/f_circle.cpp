#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "sensemaker.h"


#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif

double MinEnclosingCircle::calculate_diam(std::vector<Pixel2>& Contour)
{
    //------------------------Minimum enclosing circle------------------------------------------
    //https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L166
    // Find the minimum enclosing circle of an object
    Point2f center;
    float radius = 0;
    minEnclosingCircle(Contour, center, radius);
    //Diameter of the minimum circumference of the projection area.
    double diameter_min_enclosing_circle = 2 * radius;
    return diameter_min_enclosing_circle; // ratios[45] = diameter_min_enclosing_circle;
}

void MinEnclosingCircle::findCircle3pts (const std::vector<Pixel2>& pts, Point2f& center, float& radius)
{
    // two edges of the triangle v1, v2
    Pixel2 v1 = pts[1] - pts[0];
    Pixel2 v2 = pts[2] - pts[0];

    // center is intersection of midperpendicular lines of the two edges v1, v2
    // a1*x + b1*y = c1 where a1 = v1.x, b1 = v1.y
    // a2*x + b2*y = c2 where a2 = v2.x, b2 = v2.y

    auto midPoint1 = (pts[0] + pts[1]) / 2.0f;
    float c1 = midPoint1.x * v1.x + midPoint1.y * v1.y;
    Point2f midPoint2 = (pts[0] + pts[2]) / 2.0f;
    float c2 = midPoint2.x * v2.x + midPoint2.y * v2.y;
    float det = v1.x * v2.y - v1.y * v2.x;
    if (fabs(det) <= EPS)
    {
        // v1 and v2 are colinear, so the longest distance between any 2 points
        // is the diameter of the minimum enclosing circle.
        float d1 = normL2 (pts[0] - pts[1]);  //normL2Sqr<float>(pts[0] - pts[1]);
        float d2 = normL2 (pts[0] - pts[2]);   //normL2Sqr<float>(pts[0] - pts[2]);
        float d3 = normL2 (pts[1] - pts[2]);   //normL2Sqr<float>(pts[1] - pts[2]);
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
            //CV_DbgAssert(d3 >= d1 && d3 >= d2);
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

void MinEnclosingCircle::findThirdPoint (const std::vector<Pixel2>& pts, int i, int j, Point2f& center, float& radius)
{
    center.x = (float)(pts[j].x + pts[i].x) / 2.0f;
    center.y = (float)(pts[j].y + pts[i].y) / 2.0f;
    float dx = (float)(pts[j].x - pts[i].x);
    float dy = (float)(pts[j].y - pts[i].y);
    radius = (float)normL2(Point2f(dx, dy)) / 2.0f + EPS;

    for (int k = 0; k < j; ++k)
    {
        dx = center.x - (float)pts[k].x;
        dy = center.y - (float)pts[k].y;
        if (normL2(Point2f(dx, dy)) < radius)
        {
            continue;
        }
        else
        {
            std::vector<Pixel2> ptsf;   //Point2f ptsf[3];
            ptsf.push_back (Pixel2(pts[i].x, pts[i].y, 0));    //(Point2f)pts[i];
            ptsf.push_back(Pixel2(pts[j].x, pts[j].y, 0));   //(Point2f)pts[j];
            ptsf.push_back(Pixel2(pts[k].x, pts[k].y, 0));  //(Point2f)pts[k];
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

void MinEnclosingCircle::findSecondPoint (const std::vector<Pixel2>& pts, int i, Point2f& center, float& radius)
{
    center.x = (float)(pts[0].x + pts[i].x) / 2.0f;
    center.y = (float)(pts[0].y + pts[i].y) / 2.0f;
    float dx = (float)(pts[0].x - pts[i].x);
    float dy = (float)(pts[0].y - pts[i].y);
    radius = (float)normL2(Point2f(dx, dy)) / 2.0f + EPS;

    for (int j = 1; j < i; ++j)
    {
        dx = center.x - (float)pts[j].x;
        dy = center.y - (float)pts[j].y;
        if (normL2(Point2f(dx, dy)) < radius)
        {
            continue;
        }
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


void MinEnclosingCircle::findMinEnclosingCircle (const std::vector<Pixel2>& pts, int count, Point2f& center, float& radius)
{
    center.x = (float)(pts[0].x + pts[1].x) / 2.0f;
    center.y = (float)(pts[0].y + pts[1].y) / 2.0f;
    float dx = (float)(pts[0].x - pts[1].x);
    float dy = (float)(pts[0].y - pts[1].y);
    radius = (float)normL2(Point2f(dx, dy)) / 2.0f + EPS;

    for (int i = 2; i < count; ++i)
    {
        dx = (float)pts[i].x - center.x;
        dy = (float)pts[i].y - center.y;
        float d = (float)normL2(Point2f(dx, dy));
        if (d < radius)
        {
            continue;
        }
        else
        {
            Point2f new_center; float new_radius = 0;
            findSecondPoint(pts, i, new_center, new_radius);
            if (new_radius > 0)
            {
                radius = new_radius;
                center = new_center;
            }
        }
    }
}

// see Welzl, Emo. Smallest enclosing disks (balls and ellipsoids). Springer Berlin Heidelberg, 1991.
void MinEnclosingCircle::minEnclosingCircle(
	// in:
	std::vector<Pixel2>& Contour,
	// out:
	Point2f & _center, float& _radius)
{
#if 0
    CV_INSTRUMENT_REGION();

    Mat points = _points.getMat();
    int count = points.checkVector(2);
    int depth = points.depth();
    CV_Assert(count >= 0 && (depth == CV_32F || depth == CV_32S));
#endif


    _center.x = _center.y = 0.f;
    _radius = 0.f;
    auto count = Contour.size();

    if (count == 0)
        return;


    //bool is_float = depth == CV_32F;
    //const Point* ptsi = points.ptr<Point>();
    //const Point2f* ptsf = points.ptr<Point2f>();

    switch (count)
    {
    case 1:
    {
        _center = Point2f (Contour[0].x, Contour[0].y);   //(is_float) ? ptsf[0] : Point2f((float)ptsi[0].x, (float)ptsi[0].y);
        _radius = EPS;
        break;
    }
    case 2:
    {
        Point2f p1 = Point2f (Contour[0].x, Contour[0].y);   //(is_float) ? ptsf[0] : Point2f((float)ptsi[0].x, (float)ptsi[0].y);
        Point2f p2 = Point2f (Contour[1].x, Contour[1].y);    //(is_float) ? ptsf[1] : Point2f((float)ptsi[1].x, (float)ptsi[1].y);
        _center.x = (p1.x + p2.x) / 2.0f;
        _center.y = (p1.y + p2.y) / 2.0f;
        _radius = (float)(normL2(p1 - p2) / 2.0) + EPS;
        break;
    }
    default:
    {
        Point2f center;
        float radius = 0.f;
#if 0//andre
        if (is_float)
        {
            findMinEnclosingCircle<Point2f>(ptsf, count, center, radius);
#if 0
            for (size_t m = 0; m < count; ++m)
            {
                float d = (float)norm(ptsf[m] - center);
                if (d > radius)
                {
                    printf("error!\n");
                }
            }
#endif
        }
        else
        {
            findMinEnclosingCircle<Point>(ptsi, count, center, radius);
#if 0
            for (size_t m = 0; m < count; ++m)
            {
                double dx = ptsi[m].x - center.x;
                double dy = ptsi[m].y - center.y;
                double d = std::sqrt(dx * dx + dy * dy);
                if (d > radius)
                {
                    printf("error!\n");
                }
            }
#endif
        }
#endif//andre
        findMinEnclosingCircle (Contour, count, center, radius);
        _center = center;
        _radius = radius;
        break;
    }
    }
}


std::tuple <double, double> InscribingCircumscribingCircle::calculateInsCir (std::vector<Pixel2> & contours, double xCentroid, double yCentroid)
{
    //-----------------circumscribing and inscribing circle ---------------------------
    //https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L199

    double yCentroid2 = yCentroid - 1;
    double xCentroid2 = xCentroid - 1;
    std::vector <double> distances;

    for (size_t j = 0; j < contours.size(); j++) {
        double tmpx = (contours[j].x - xCentroid2);
        double tmpy = (contours[j].y - yCentroid2);
        double distance = sqrt(tmpx * tmpx + tmpy * tmpy);
        distances.push_back(distance);
    }

    double radius_circumscribing_circle = *std::max_element(distances.begin(), distances.end());
    double radius_inscribing_circle = *std::min_element(distances.begin(), distances.end());

    double diameter_circumscribing_circle = 2 * radius_circumscribing_circle;
    double diameter_inscribing_circle = 2 * radius_inscribing_circle;

    //ratios[46] = diameter_circumscribing_circle;
    //ratios[47] = diameter_inscribing_circle;
    return { diameter_inscribing_circle, diameter_circumscribing_circle };
}

