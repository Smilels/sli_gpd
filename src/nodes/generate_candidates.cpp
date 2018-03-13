// ROS
#include <ros/ros.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
// Custom
#include <gpg/cloud_camera.h>
#include <gpg/candidates_generator.h>
#include <gpg/plot.h>

int main(int argc, char* argv[])
{
  // initialize ROS
  ros::init(argc, argv, "generate_grasp_candidates");
  ros::NodeHandle node("~");

  // Create objects to store parameters
  CandidatesGenerator::Parameters generator_params;
  HandSearch::Parameters hand_search_params;
  Plot plotter;

  // Read hand geometry parameters
  node.param("finger_width", hand_search_params.finger_width_, 0.01);
  node.param("hand_outer_diameter", hand_search_params.hand_outer_diameter_, 0.09);
  node.param("hand_depth", hand_search_params.hand_depth_, 0.06);
  node.param("hand_height", hand_search_params.hand_height_, 0.02);
  node.param("init_bite", hand_search_params.init_bite_, 0.015);

  // Read local hand search parameters
  node.param("nn_radius", hand_search_params.nn_radius_frames_, 0.01);
  node.param("num_orientations", hand_search_params.num_orientations_, 8);
  node.param("num_samples", hand_search_params.num_samples_, 500);
  node.param("num_threads", hand_search_params.num_threads_, 1);
  node.param("rotation_axis", hand_search_params.rotation_axis_, 2);

  // Read general parameters
  generator_params.num_samples_ = hand_search_params.num_samples_;
  generator_params.num_threads_ = hand_search_params.num_threads_;
  node.param("plot_candidates", generator_params.plot_grasps_, false);

  // Read preprocessing parameters
  node.param("remove_outliers", generator_params.remove_statistical_outliers_, true);
  node.param("voxelize", generator_params.voxelize_, true);
  node.getParam("workspace", generator_params.workspace_);
  bool downward_filter_;
  node.param("downward_filter", downward_filter_,true);
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);

  // Set the position from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3,1);
  view_points << camera_position[0], camera_position[1], camera_position[2];

  // Load point cloud from file
  std::string filename;
  node.param("cloud_file_name", filename, std::string(""));
  CloudCamera cloud_cam(filename, view_points);
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Create object to generate grasp candidates.
  CandidatesGenerator candidates_generator(generator_params, hand_search_params);

  // Preprocess the point cloud: voxelization, removing statistical outliers, workspace filtering.
  candidates_generator.preprocessPointCloud(cloud_cam);

  // Generate grasp candidates.
  std::vector<Grasp> candidates = candidates_generator.generateGraspCandidates(cloud_cam);

  const HandSearch::Parameters& params = candidates_generator.getHandSearchParams();
  plotter.plotFingers3D(candidates, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
    params.finger_width_, params.hand_depth_, params.hand_height_);


  if (downward_filter_)
  {
    std::vector<Grasp> val_hands;
    std::cout<<"use downward_filter_"<<std::endl;
    //listen to the transform, in order to transfer the vector
    //the transform from frame /table_top to frame kinect2_rgb_optical_frame.
    //tf::StampedTransform transform;
    // try{
    //   tf_listener->waitForTransform("kinect2_rgb_optical_frame","/table_top", ros::Time::now(),ros::Duration(5.0));
    //   tf_listener->lookupTransform ("kinect2_rgb_optical_frame","/table_top", ros::Time(0), transform);
    // }
    // catch(std::runtime_error &e){
    //   std::cout<<"tf listener between kinect2 and table_top happens error"<<std::endl;
    //   return selected_grasps;
    // }

    // tf::Matrix3x3 uptf;
    // uptf.setRotation(transform.inverse().getRotation());
    // Eigen::Matrix3d trans;
    // tf::matrixTFToEigen(uptf,trans);

    //remedy invaild grasps
      val_hands=candidates;
      for (int j = 0; j < val_hands.size(); j++)
      {
        Eigen::Matrix3d val_frame=val_hands[j].getFrame();
        //Eigen::Matrix3d val_frame=trans*frame_rot;// frame represents in table_top

        //calculate the angle between upright direction and approach direction
        tf::Vector3 cam_approch;
        tf::vectorEigenToTF(val_frame.col(0),cam_approch);
        tf::Vector3 cam_z=tf::Vector3 (0,0,1);
        tfScalar up_angle=cam_approch.angle (cam_z);
        if (up_angle*180/M_PI<90)
        {
          Eigen::Matrix3d frame_mat;
          frame_mat=val_frame;
          frame_mat.col(0)<<val_frame.col(0)(0),val_frame.col(0)(1),0;
          frame_mat.col(2)=frame_mat.col(0).cross(frame_mat.col(1));
          //val_hands[j].pose_.frame_=trans.inverse()*frame_mat; //frame transfer back
          val_hands[j].pose_.frame_=frame_mat;
        }
      }
      if (generator_params.plot_grasps_)
      {
        plotter.plotFingers3D(val_hands, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
          params.finger_width_, params.hand_depth_, params.hand_height_);
      }
    }



  return 0;
}
