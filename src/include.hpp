/*
 * include.hpp
 *
 *  Created on: Dec 5, 2013
 *      Author: jose
 */

#ifndef INCLUDE_HPP_
#define INCLUDE_HPP_

#include <numeric>

#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/common/angles.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/io/pcd_io.h>

using pcl::PointXYZRGBNormal;
using pcl::PointXYZRGB;
//XYZ and RGB Pointcloud

typedef pcl::PointCloud<PointXYZRGB> PCXYZRGB;
typedef PCXYZRGB::Ptr PCXYZRGBPtr;

//XYZ and normal Pointcloud
typedef pcl::PointCloud<PointXYZRGBNormal> PCNormal;
typedef PCNormal::Ptr PCNormalPtr;

typedef pcl::FPFHSignature33 FPFHSignature;
typedef pcl::PointCloud<pcl::FPFHSignature33> PCFPFH;
typedef PCFPFH::Ptr PCFPFHPtr;

typedef boost::shared_ptr<Eigen::Matrix4d> transformPtr;

#endif /* INCLUDE_HPP_ */
