/*Initial Project*/

#include <iostream>
#include <pcl/visualization/cloud_viewer.h>
#include <math.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/console/time.h>
#include <pcl/filters/normal_space.h>
#include <pcl/correspondence.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "include.hpp"

#include <pcl/features/integral_image_normal.h>
#include <boost/filesystem.hpp>
#include <pcl/filters/extract_indices.h>

typedef std::vector< pcl::Correspondence, Eigen::aligned_allocator<pcl::Correspondence> > Correspondences;
typedef boost::shared_ptr<Correspondences> CorrespondencesPtr;

void visualizeCorrespondances(const PCXYZRGBPtr &cloud_src
		, const PCXYZRGBPtr &cloud_tgt
		, const PCNormalPtr &keypoints_src
		, const PCNormalPtr &keypoints_tgt
		, const CorrespondencesPtr &corr)
{
	//this is the visualizer
	pcl::visualization::PCLVisualizer viscorr;
	viscorr.addPointCloud(cloud_src, "src_points");
	viscorr.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,0.5,"src_points");

	pcl::visualization::PointCloudColorHandlerCustom<PointXYZRGBNormal> kpoint_color_handler (keypoints_src, 0, 255, 0);
	viscorr.addPointCloud<PointXYZRGBNormal>(keypoints_src, kpoint_color_handler, "keypoints1");
	viscorr.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "keypoints1");

	Eigen::Matrix4f t;
	t<<1,0,0,0,
			0,1,0,0,
			0,0,1,0,
			0,0,0,1;

	//cloud view contains the translated cloud
	PCXYZRGBPtr cloudview(new PCXYZRGB);
	PCNormalPtr keypointsView(new PCNormal);
	//cloudNext is my target cloud
	pcl::transformPointCloud(*cloud_tgt,*cloudview,t);
	pcl::transformPointCloud(*keypoints_tgt,*keypointsView,t);

	viscorr.addPointCloud(cloudview, "tgt_points");
	viscorr.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,0.5,"tgt_points");

	pcl::visualization::PointCloudColorHandlerCustom<PointXYZRGBNormal> kpoint_color_handler2 (keypointsView, 0, 255, 0);
	viscorr.addPointCloud<PointXYZRGBNormal>(keypointsView, kpoint_color_handler2, "keypoints2");
	viscorr.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "keypoints2");

	viscorr.addCorrespondences<PointXYZRGBNormal>(keypoints_src,keypointsView,*corr,1,"Correspondences");
	viscorr.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "Correspondences");
	viscorr.resetCamera ();
	viscorr.spin ();
}

int main (int cargs, char** vargs)
{

	if(cargs<3)
	{
		std::cout<<"Filename Required"<<std::endl
				<<"Example:"<<std::endl
				<<"visualizer pointcloud.pcd"<<std::endl;
	}

	pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

	PCXYZRGBPtr target (new PCXYZRGB);
	PCXYZRGBPtr origin (new PCXYZRGB);

	if(pcl::io::loadPCDFile (vargs[1], *target)!=0)
	{
		std::cout<<"Error Reading the file, Invalid or corrupted"<<std::endl;
		return 0;
	}

	if(pcl::io::loadPCDFile (vargs[2], *origin)!=0)
	{
		std::cout<<"Error Reading the file, Invalid or corrupted"<<std::endl;
		return 0;
	}

	pcl::console::TicToc timer;
	timer.tic();

	/*********************************filtering*******************************************/
	pcl::FastBilateralFilterOMP<pcl::PointXYZRGB> bilateral_filter;
	bilateral_filter.setSigmaS (5);
	bilateral_filter.setSigmaR (0.05f);

	PCXYZRGBPtr filteredTarget (new PCXYZRGB);
	bilateral_filter.setInputCloud (target);
	bilateral_filter.filter(*filteredTarget);

	PCXYZRGBPtr filteredOrigin (new PCXYZRGB);
	bilateral_filter.setInputCloud (origin);
	bilateral_filter.filter(*filteredOrigin);

	/********************************Normals***************************************************/
	pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.01f);
	ne.setNormalSmoothingSize(10.0f);

	pcl::PointCloud<pcl::Normal>::Ptr targetNormals (new pcl::PointCloud<pcl::Normal>);
	PCNormalPtr targetCloudNormals (new PCNormal);
	ne.setInputCloud(filteredTarget);
	ne.compute(*targetNormals);
	pcl::concatenateFields (*filteredTarget, *targetNormals, *targetCloudNormals);

	pcl::PointCloud<pcl::Normal>::Ptr originNormals (new pcl::PointCloud<pcl::Normal>);
	PCNormalPtr originCloudNormals (new PCNormal);
	ne.setInputCloud(filteredOrigin);
	ne.compute(*originNormals);
	pcl::concatenateFields (*filteredOrigin, *originNormals, *originCloudNormals);

	/**********************************************Remove biggest plane***************************************/
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	PCNormalPtr originCloudNormalsPR (new PCNormal);
	PCNormalPtr targetCloudNormalsPR (new PCNormal);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;
	// Optional
	seg.setOptimizeCoefficients (true);
	// Mandatory
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setDistanceThreshold (0.05);//0.05

	pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
	extract.setNegative (true);

	seg.setInputCloud (originCloudNormals);
	seg.segment(*inliers, *coefficients);

	extract.setInputCloud (originCloudNormals);
	extract.setIndices (inliers);
	extract.filter (*originCloudNormalsPR);

	seg.setInputCloud (targetCloudNormals);
	seg.segment(*inliers, *coefficients);

	extract.setInputCloud (targetCloudNormals);
	extract.setIndices (inliers);
	extract.filter (*targetCloudNormalsPR);

	cout<<"Sampling"<<endl;
	/**********************************************Sampling**************************************************/
	pcl::NormalSpaceSampling<PointXYZRGBNormal,PointXYZRGBNormal> normal_sampling;
	normal_sampling.setBins (15, 15, 15);

	int minsize = !(targetCloudNormalsPR->size () / 8<originCloudNormalsPR->size () / 8)?originCloudNormalsPR->size () / 8:targetCloudNormalsPR->size () / 8;
	int samplesize = (minsize>8000) ? minsize :8000;
	normal_sampling.setSample (samplesize);

	PCNormalPtr sampledTarget (new PCNormal);
	normal_sampling.setInputCloud (targetCloudNormalsPR);
	normal_sampling.setNormals (targetCloudNormalsPR);
	normal_sampling.filter (*sampledTarget);

	PCNormalPtr sampledOrigin (new PCNormal);
	normal_sampling.setInputCloud (originCloudNormalsPR);
	normal_sampling.setNormals (originCloudNormalsPR);
	normal_sampling.filter (*sampledOrigin);

	cout<<endl<<"target original size "<<targetCloudNormalsPR->size ()<<" Sampling size "<<sampledTarget->size ()<<endl;
	cout<<endl<<"Origin original size "<<originCloudNormalsPR->size ()<<" Sampling size "<<sampledOrigin->size ()<<endl;
	/***************************************ICP******************************************************************/

	Eigen::Matrix4d transform (Eigen::Matrix4d::Identity ());
	Eigen::Matrix4d final_transform (Eigen::Matrix4d::Identity ());
	CorrespondencesPtr all_correspondences (new Correspondences),
			good_correspondences (new Correspondences);
	PCNormalPtr output (new PCNormal);
	PCXYZRGBPtr registeredOrigin (new PCXYZRGB);
	pcl::console::TicToc timer2;
	pcl::console::TicToc timer3;



	int iterations = 0;
	// Convergence criteria class stops the loop
	pcl::registration::DefaultConvergenceCriteria<double> converged (iterations, transform, *good_correspondences);

	// ICP loop
	do
	{
		timer2.tic();
		timer3.tic();
		transformPointCloudWithNormals (*sampledOrigin, *output, final_transform.cast<float> ());
//		if((iterations%9-1)==0)
//						{
//							transformPointCloud (*origin, *registeredOrigin, final_transform.cast<float> ());
//							visualizeCorrespondances(registeredOrigin,target,output,sampledTarget,good_correspondences);
//						}
		cout<<"Transform clouds "<<timer3.toc()<<"ms"<<endl;

		// Find correspondences
		timer3.tic();
		pcl::registration::CorrespondenceEstimationBackProjection<PointXYZRGBNormal, PointXYZRGBNormal, PointXYZRGBNormal> est;
		est.setInputSource (output);
		est.setInputTarget (sampledTarget);

		est.setSourceNormals (output);
		est.setTargetNormals (sampledTarget); //Used on BackProjectrion
		est.setKSearch (20); //Also try with 30

		// Reciprocal determines if the correspondence chosen is reciprocal.
		est.determineReciprocalCorrespondences (*all_correspondences);
		cout<<"Correspondance estimation "<<timer3.toc()<<"ms"<<endl;


		//		Visualize first iteration all correspondances
		//		if(iterations==0)
		//			visualizeCorrespondances(output,tgt,output,tgt,all_correspondences);
		//		PCL_DEBUG ("Number of correspondences found: %d\n", all_correspondences->size ());

		// Reject correspondences

		timer3.tic();
		pcl::registration::CorrespondenceRejectorMedianDistance rej;
		rej.setMedianFactor (4.0f);
		rej.setInputCorrespondences (all_correspondences);
		rej.getCorrespondences (*good_correspondences);
		cout<<"Correspondance rejection "<<timer3.toc()<<"ms"<<endl;
		//cout<<"Reject Correspondances "<<": "<<timer.toc()<<" Miliseconds"<<endl;
		//			PCL_DEBUG ("Number of correspondences remaining after rejection: %d\n", good_correspondences->size ());

		//				Visualize first iteration good correspondances
		//		if(iterations==0)
		//					visualizeCorrespondances(output,tgt,output,tgt,good_correspondences);

		// Find transformation
		timer3.tic();
		pcl::registration::TransformationEstimationPointToPlaneLLS<PointXYZRGBNormal, PointXYZRGBNormal, double> trans_est;
		trans_est.estimateRigidTransformation (*output, *sampledTarget, *good_correspondences, transform);
		//cout<<"Find Transformation "<<": "<<timer.toc()<<" Miliseconds"<<endl;
		// Obtain the final transformation
		final_transform = transform * final_transform;
		cout<<"Find transform "<<timer3.toc()<<"ms"<<endl;


		// Transform the data
		//cout<<"Transform Cloud "<<": "<<timer.toc()<<" Miliseconds"<<endl;
		// Check if convergence has been reached

		++iterations;
		cout<<"Iteration "<<iterations<<" took "<<timer2.toc()<<"ms"<<endl;


	}
	while (!converged);


	transformPointCloud (*origin, *registeredOrigin, final_transform.cast<float> ());

	cout << "Converged at "<<iterations<<" iterations"<<endl;
	//	cout<<"Converged "<<converged<<endl;
	//	view (src, tgt, good_correspondences);
	//	view (output, tgt, good_correspondences);



	cout<< "Function took "<<timer.toc()<<"ms"<<endl;

	pcl::visualization::Camera cam;
	cam.clip[0]=2.90025;
	cam.clip[1] = 20.0679;
	cam.focal[0] = 0.398347;
	cam.focal[1] = -0.11088;
	cam.focal[2] = 2.156;
	cam.pos[0] = 1.50747;
	cam.pos[1] = -0.889271;
	cam.pos[2] = -5.61117;
	cam.view[0] = -0.0836109;
	cam.view[1] = -0.992646;
	cam.view[2] = 0.0875393;
	cam.fovy = 0.523599;
	cam.window_size[0] = 1600;
	cam.window_size[1] = 600;
	cam.window_pos[0] = 359;
	cam.window_pos[1] = 252;

	int v1,v2;
	int v3,v4;
	int v5,v6;
	int v7,v8;
	int v9,v10;


	pcl::visualization::PCLVisualizer viewer5("Filtered Plane removed");
	viewer5.createViewPort(0,0,0.5,1,v9);
	viewer5.createViewPort(0.5,0,1,1,v10);
	viewer5.setCameraParameters(cam);
	pcl::visualization::PointCloudColorHandlerRGBField<PointXYZRGBNormal> handler6(targetCloudNormalsPR);
	viewer5.addPointCloud(targetCloudNormalsPR,handler6,"CloudFinal",v9);
	viewer5.addPointCloudNormals<PointXYZRGBNormal,PointXYZRGBNormal>(targetCloudNormalsPR,targetCloudNormalsPR,1,0.05,"Target2",v9);
	pcl::visualization::PointCloudColorHandlerRGBField<PointXYZRGBNormal> handler5(originCloudNormalsPR);
	viewer5.addPointCloud(originCloudNormalsPR,handler5,"CloudFinal2",v10);
	viewer5.addPointCloudNormals<PointXYZRGBNormal,PointXYZRGBNormal>(originCloudNormalsPR,originCloudNormalsPR,1,0.05,"origin2",v10);


	pcl::visualization::PCLVisualizer viewer("Filtered");
	viewer.createViewPort(0,0,0.5,1,v1);
	viewer.createViewPort(0.5,0,1,1,v2);
	viewer.setCameraParameters(cam);
	viewer.addPointCloud(filteredTarget,"Target",v1);
	viewer.addPointCloud(filteredOrigin,"Origin",v2);


	pcl::visualization::PCLVisualizer viewer2("Filtered");
	viewer2.createViewPort(0,0,0.5,1,v3);
	viewer2.createViewPort(0.5,0,1,1,v4);
	viewer2.setCameraParameters(cam);
	pcl::visualization::PointCloudColorHandlerRGBField<PointXYZRGBNormal> handler(targetCloudNormals);
	viewer2.addPointCloud(targetCloudNormals,handler,"CloudFinal",v3);
	viewer2.addPointCloudNormals<PointXYZRGBNormal,PointXYZRGBNormal>(targetCloudNormals,targetCloudNormals,5,0.05,"Target2",v3);
	pcl::visualization::PointCloudColorHandlerRGBField<PointXYZRGBNormal> handler2(originCloudNormals);
	viewer2.addPointCloud(originCloudNormals,handler2,"CloudFinal2",v4);
	viewer2.addPointCloudNormals<PointXYZRGBNormal,PointXYZRGBNormal>(originCloudNormals,originCloudNormals,5,0.05,"origin2",v4);

	pcl::visualization::PCLVisualizer viewer3("Sub Sampled");
	viewer3.createViewPort(0,0,0.5,1,v5);
	viewer3.createViewPort(0.5,0,1,1,v6);
	viewer3.setCameraParameters(cam);
	pcl::visualization::PointCloudColorHandlerRGBField<PointXYZRGBNormal> handler3(sampledTarget);
	viewer3.addPointCloud(sampledTarget,handler2,"CloudFinal3",v5);
	viewer3.addPointCloudNormals<PointXYZRGBNormal,PointXYZRGBNormal>(sampledTarget,sampledTarget,1,0.07,"t3",v5);

	pcl::visualization::PointCloudColorHandlerRGBField<PointXYZRGBNormal> handler4(sampledOrigin);
	viewer3.addPointCloud(sampledOrigin,handler4,"CloudFinal4",v6);
	viewer3.addPointCloudNormals<PointXYZRGBNormal,PointXYZRGBNormal>(sampledOrigin,sampledOrigin,1,0.07,"o3",v6);


	pcl::visualization::PCLVisualizer viewer4("Registered");
	viewer4.createViewPort(0,0,0.5,1,v7);
	viewer4.createViewPort(0.5,0,1,1,v8);
	viewer4.setCameraParameters(cam);
	viewer4.addPointCloud(target,"Targetn",v7);
	viewer4.addPointCloud(origin,"originn",v7);
	viewer4.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,0.7f,"originn",v7);

	viewer4.addPointCloud(target,"targetn2",v8);
	viewer4.addPointCloud(registeredOrigin,"originr",v8);
	viewer4.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,0.7f,"originr",v8);


	//blocks until the cloud is actually rendered
	viewer4.spin();
	cout<<endl<<endl<<"-------------------------Finished---------------------------";
	return 0;
}

