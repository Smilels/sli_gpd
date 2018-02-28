#include "../../include/gpd/grasp_detector.h"


GraspDetector::GraspDetector(ros::NodeHandle& node)
{
  Eigen::initParallel();

  // Create objects to store parameters.
  CandidatesGenerator::Parameters generator_params;
  HandSearch::Parameters hand_search_params;

  // Read hand geometry parameters.
  node.param("finger_width", hand_search_params.finger_width_, 0.01);
  node.param("hand_outer_diameter", hand_search_params.hand_outer_diameter_, 0.09);
  node.param("hand_depth", hand_search_params.hand_depth_, 0.06);
  node.param("hand_height", hand_search_params.hand_height_, 0.02);
  node.param("init_bite", hand_search_params.init_bite_, 0.015);
  outer_diameter_ = hand_search_params.hand_outer_diameter_;

  // Read local hand search parameters.
  node.param("nn_radius", hand_search_params.nn_radius_frames_, 0.01);
  node.param("num_orientations", hand_search_params.num_orientations_, 8);
  node.param("num_samples", hand_search_params.num_samples_, 500);
  node.param("num_threads", hand_search_params.num_threads_, 1);
  node.param("rotation_axis", hand_search_params.rotation_axis_, 2); // cannot be changed

  // Read plotting parameters.
  node.param("plot_samples", plot_samples_, false);
  node.param("plot_normals", plot_normals_, false);
  generator_params.plot_normals_ = plot_normals_;
  node.param("plot_filtered_grasps", plot_filtered_grasps_, false);
  node.param("plot_valid_grasps", plot_valid_grasps_, false);
  node.param("plot_clusters", plot_clusters_, false);
  node.param("plot_selected_grasps", plot_selected_grasps_, false);

  // Read general parameters.
  generator_params.num_samples_ = hand_search_params.num_samples_;
  generator_params.num_threads_ = hand_search_params.num_threads_;
  node.param("plot_candidates", generator_params.plot_grasps_, false);

  // Read preprocessing parameters.
  node.param("remove_outliers", generator_params.remove_statistical_outliers_, true);
  node.param("voxelize", generator_params.voxelize_, true);
  node.getParam("workspace", generator_params.workspace_);
  node.getParam("workspace_grasps", workspace_);

  // Create object to generate grasp candidates.
  candidates_generator_ = new CandidatesGenerator(generator_params, hand_search_params);

  // Read classification parameters and create classifier.
  std::string model_file, weights_file;
  bool use_gpu;
  node.param("model_file", model_file, std::string(""));
  node.param("trained_file", weights_file, std::string(""));
  node.param("min_score_diff", min_score_diff_, 500.0);
  node.param("create_image_batches", create_image_batches_, true);
  node.param("use_gpu", use_gpu, true);
  classifier_ = new CaffeClassifier(model_file, weights_file, use_gpu);

  // Read grasp image parameters.
  node.param("image_outer_diameter", image_params_.outer_diameter_, hand_search_params.hand_outer_diameter_);
  node.param("image_depth", image_params_.depth_, hand_search_params.hand_depth_);
  node.param("image_height", image_params_.height_, hand_search_params.hand_height_);
  node.param("image_size", image_params_.size_, 60);
  node.param("image_num_channels", image_params_.num_channels_, 15);

  // Read learning parameters.
  bool remove_plane;
  node.param("remove_plane_before_image_calculation", remove_plane, false);

  // Create object to create grasp images from grasp candidates (used for classification)
  learning_ = new Learning(image_params_, hand_search_params.num_threads_, hand_search_params.num_orientations_, false, remove_plane);

  // Read grasp filtering parameters
  node.param("filter_grasps", filter_grasps_, false);
  node.param("filter_half_antipodal", filter_half_antipodal_, false);
  std::vector<double> gripper_width_range(2);
  gripper_width_range[0] = 0.03;
  gripper_width_range[1] = 0.07;
  node.getParam("gripper_width_range", gripper_width_range);
  min_aperture_ = gripper_width_range[0];
  max_aperture_ = gripper_width_range[1];

  // Read clustering parameters
  int min_inliers;
  node.param("min_inliers", min_inliers, 0);
  clustering_ = new Clustering(min_inliers);
  cluster_grasps_ = min_inliers > 0 ? true : false;

  // Read grasp selection parameters
  node.param("num_selected", num_selected_, 100);
}


std::vector<Grasp> GraspDetector::detectGrasps(const CloudCamera& cloud_cam)
{
  std::vector<Grasp> selected_grasps(0);

  // Check if the point cloud is empty.
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    ROS_INFO("Point cloud is empty!");
    return selected_grasps;
  }

  Plot plotter;

  // Plot samples/indices.
  if (plot_samples_)
  {
    if (cloud_cam.getSamples().cols() > 0)
    {
      plotter.plotSamples(cloud_cam.getSamples(), cloud_cam.getCloudProcessed());
    }
    else if (cloud_cam.getSampleIndices().size() > 0)
    {
      plotter.plotSamples(cloud_cam.getSampleIndices(), cloud_cam.getCloudProcessed());
    }
  }

  if (plot_normals_)
  {
    std::cout << "Plotting normals for different camera sources\n";
    plotter.plotNormals(cloud_cam);
  }

  // 1. Generate grasp candidates.
  std::vector<GraspSet> candidates = generateGraspCandidates(cloud_cam);
  ROS_INFO_STREAM("Generated " << candidates.size() << " grasp candidate sets.");
  if (candidates.size() == 0)
  {
    return selected_grasps;
  }

  // 2.1 Prune grasp candidates based on min. and max. robot hand aperture and fingers below table surface.
  if (filter_grasps_)
  {
    candidates = filterGraspsWorkspace(candidates, workspace_);

    if (plot_filtered_grasps_)
    {
      plotter.plotFingers(candidates, cloud_cam.getCloudProcessed(), "Filtered Grasps");
    }
  }

  // 2.2 Filter half grasps.
  if (filter_half_antipodal_)
  {
    candidates = filterHalfAntipodal(candidates);

    if (plot_filtered_grasps_)
    {
      plotter.plotFingers(candidates, cloud_cam.getCloudProcessed(), "Filtered Grasps");
    }
  }

  // 3. Classify each grasp candidate. (Note: switch from a list of hypothesis sets to a list of grasp hypotheses)
  std::vector<Grasp> valid_grasps = GMM_classifyGraspCandidates(cloud_cam, candidates);
  ROS_INFO_STREAM("Predicted " << valid_grasps.size() << " valid grasps.");

  if (valid_grasps.size() <= 2)
  {
    std::cout << "Not enough valid grasps predicted! Using all grasps from previous step.\n";
    valid_grasps = extractHypotheses(candidates);
  }

  // 4. Cluster the grasps.
  std::vector<Grasp> clustered_grasps;
  clustered_grasps = valid_grasps;

  // 5. Select highest-scoring grasps.
  if (clustered_grasps.size() > num_selected_)
  {
    std::cout << "Partial Sorting the grasps based on their score ... \n";
    std::partial_sort(clustered_grasps.begin(), clustered_grasps.begin() + num_selected_, clustered_grasps.end(),
      isScoreGreater);
    selected_grasps.assign(clustered_grasps.begin(), clustered_grasps.begin() + num_selected_);
  }
  else
  {
    std::cout << "Sorting the grasps based on their score ... \n";
    std::sort(clustered_grasps.begin(), clustered_grasps.end(), isScoreGreater);
    selected_grasps = clustered_grasps;
    //selected_grasps.push_back(clustered_grasps[0]);
  }

  for (int i = 0; i < selected_grasps.size(); i++)
  {
    std::cout << "Grasp " << i << ": " << selected_grasps[i].getScore() << "\n";
  }

  ROS_INFO_STREAM("Selected the " << selected_grasps.size() << " highest scoring grasps.");

  if (plot_selected_grasps_)
  {
    const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
    plotter.plotFingers3D(selected_grasps, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
      params.finger_width_, params.hand_depth_, params.hand_height_);
  }

  // 3. Classify each grasp candidate. (Note: switch from a list of hypothesis sets to a list of grasp hypotheses)
  std::vector<Grasp> valid_grasps1 = classifyGraspCandidates(cloud_cam, candidates);
  ROS_INFO_STREAM("orignal Predicted " << valid_grasps1.size() << " valid grasps.");

  if (valid_grasps1.size() <= 2)
  {
    std::cout << "Not enough valid grasps predicted! Using all grasps from previous step.\n";
    valid_grasps1 = extractHypotheses(candidates);
  }

  // 4. Cluster the grasps.
  std::vector<Grasp> clustered_grasps1;

  if (cluster_grasps_)
  {
    clustered_grasps1 = findClusters(valid_grasps1);
    ROS_INFO_STREAM("Found " << clustered_grasps1.size() << " clusters.");
    if (clustered_grasps1.size() <= 3)
    {
      std::cout << "Not enough clusters found! Using all grasps from previous step.\n";
      clustered_grasps1 = valid_grasps1;
    }

    if (plot_clusters_)
    {
      plotter.plotFingers(clustered_grasps1, cloud_cam.getCloudOriginal(), "Clustered Grasps (Original Point Cloud)");
    }
  }
  else
  {
    clustered_grasps1 = valid_grasps1;
  }

  // 5. Select highest-scoring grasps.
  if (clustered_grasps1.size() > num_selected_)
  //if (clustered_grasps.size() > 1000)
  {
    std::cout << "Partial Sorting the grasps based on their score ... \n";
    std::partial_sort(clustered_grasps1.begin(), clustered_grasps1.begin() + num_selected_, clustered_grasps1.end(),
      isScoreGreater);
    selected_grasps1.assign(clustered_grasps1.begin(), clustered_grasps1.begin() + num_selected_);
  }
  else
  {
    std::cout << "Sorting the grasps based on their score ... \n";
    std::sort(clustered_grasps1.begin(), clustered_grasps1.end(), isScoreGreater);
    selected_grasps1 = clustered_grasps1;
    //selected_grasps.push_back(clustered_grasps[0]);
  }

  for (int i = 0; i < selected_grasps1.size(); i++)
  {
    std::cout << "Grasp " << i << ": " << selected_grasps1[i].getScore() << "\n";
  }

  ROS_INFO_STREAM("Selected the " << selected_grasps1.size() << " highest scoring grasps.");

  if (plot_selected_grasps_)
  {
    const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
    plotter.plotFingers3D(selected_grasps1, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
      params.finger_width_, params.hand_depth_, params.hand_height_);
  }

  return selected_grasps;
}


std::vector<GraspSet> GraspDetector::generateGraspCandidates(const CloudCamera& cloud_cam)
{
  return candidates_generator_->generateGraspCandidateSets(cloud_cam);
}


void GraspDetector::preprocessPointCloud(CloudCamera& cloud_cam)
{
  candidates_generator_->preprocessPointCloud(cloud_cam);
}


std::vector<Grasp> GraspDetector::GMM_classifyGraspCandidates(const CloudCamera& cloud_cam,
  std::vector<GraspSet>& candidates)
{
  // Create a grasp image for each grasp candidate.
  std::vector<cv::Mat> image_list = learning_->createImages(cloud_cam, candidates);
  std::vector<Grasp> valid_grasps;
  std::vector<cv::Mat> valid_images;
  extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);
  int num_iters=2;
  Grasp elite_grasp;
  elite_grasp.setScore(0);
  double t0 = omp_get_wtime();
  for (int i=0;i<num_iters;i++)
  {
    // Classify the grasp images.
    std::vector<float> scores;
    scores = classifier_->classifyImages(valid_images);
    std::vector<Grasp> grasp_list;
    grasp_list.assign(valid_grasps.begin(), valid_grasps.end());
    std::sort( grasp_list.begin(), grasp_list.end(), isScoreGreater);//sort the grtaps based on their score
    std::vector<Grasp> elite_grasps;

    int num_grasps=grasp_list.size();
    double gmm_refit=0.25;
    int num_refit_=std::ceil(gmm_refit * num_grasps);
    int num_refit =std::max(num_refit_,1); //max #include <algorithm>
    ROS_INFO_STREAM("number use to gmm fit:"<<num_refit);
    elite_grasps.assign(grasp_list.begin(), grasp_list.begin() + num_refit);
    if (plot_valid_grasps_)
    {
      Plot plotter;
      const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
      plotter.plotFingers3D(elite_grasps, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
        params.finger_width_, params.hand_depth_, params.hand_height_);
    }

    std::vector<Grasp> clustered_grasps = findClusters(elite_grasps);
    ROS_INFO_STREAM("Found " << clustered_grasps.size() << " clusters.");
    arma::mat elite_grasps_arr;
    for (auto& g : elite_grasps){
      arma::mat elite_grasp_v;
      Eigen::Vector3d v_bottom,v_surface,v_sample;
      Eigen::Matrix3d v_frame;
      v_bottom=g.getGraspBottom();
      v_surface=g.getGraspSurface();
      v_frame=g.getFrame();
      v_sample=g.getSample();
      Eigen::MatrixXd v(v_bottom.rows(),v_bottom.cols()+v_surface.cols()+v_frame.cols()+v_sample.cols());
      v << v_bottom, v_surface, v_frame, v_sample;
      elite_grasp_v = arma::mat(v.data(), v.rows(), v.cols(),false, false);
      elite_grasp_v.reshape(elite_grasp_v.n_rows*elite_grasp_v.n_cols,1);
      elite_grasps_arr=arma::join_rows(elite_grasps_arr,elite_grasp_v);
    }

    arma::mat elite_grasps_mean=arma::mean(elite_grasps_arr,1);
    arma::mat elite_grasps_std=arma::stddev(elite_grasps_arr,0,1);
    elite_grasps_std.elem(find(elite_grasps_std==0)).ones();
    elite_grasps_arr = (elite_grasps_arr.each_col() - elite_grasps_mean).each_col()/ elite_grasps_std;

    arma::gmm_full gmm_model;//clustered_grasps.size()
    bool status = gmm_model.learn(elite_grasps_arr, 3, arma::maha_dist, arma::random_subset, 10, 8, 0.01, true);
    if (!status)
     ROS_ERROR("GaussianMixture learn");
    int num_gmm_samples=100;
    arma::mat grasp_vecs_mat=gmm_model.generate(num_gmm_samples);
    grasp_vecs_mat=(grasp_vecs_mat.each_col() % elite_grasps_std).each_col()+elite_grasps_mean;
    //arma::mat grasp_vecs_mat=(elite_grasps_arr.each_col() % elite_grasps_std).each_col()+elite_grasps_mean;

    std::vector<Grasp> gmm_candidates=from_feature_vec(grasp_vecs_mat);
    if (plot_valid_grasps_)
    {
      Plot plotter;
      const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
      plotter.plotFingers3D(gmm_candidates, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
        params.finger_width_, params.hand_depth_, params.hand_height_);
    }
    std::vector<cv::Mat> gmm_image_list = learning_->createImages(cloud_cam, gmm_candidates);
    extractGraspsAndImages(gmm_candidates, gmm_image_list, valid_grasps, valid_images);
  }
  std::vector<float> scores_end;
  scores_end = classifier_->classifyImages(valid_images);
  std::vector<Grasp> grasp_list;
  grasp_list.assign(valid_grasps.begin(), valid_grasps.end());
  std::cout << "Total classification time: " << omp_get_wtime() - t0 << std::endl;

  std::vector<Grasp> valid_grasps_end;
  for (int i = 0; i < grasp_list.size(); i++)
  {
    if (scores_end[i] >= 0)
    {
      std::cout << "grasp #" << i << ", score: " << scores_end[i] << "\n";
      valid_grasps_end.push_back(grasp_list[i]);
      valid_grasps_end[valid_grasps_end.size() - 1].setScore(scores_end[i]);
      valid_grasps_end[valid_grasps_end.size() - 1].setFullAntipodal(true);
    }
  }


  if (plot_valid_grasps_)
  {
    Plot plotter;
    const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
    plotter.plotFingers3D(valid_grasps_end, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
      params.finger_width_, params.hand_depth_, params.hand_height_);
  }
  return valid_grasps_end;
}

std::vector<Grasp> GraspDetector::classifyGraspCandidates(const CloudCamera& cloud_cam,
  std::vector<GraspSet>& candidates)
{
  // Create a grasp image for each grasp candidate.
  double t0 = omp_get_wtime();
  std::cout << "Creating grasp images for classifier input ...\n";
  std::vector<float> scores;
  std::vector<Grasp> grasp_list;
  int num_orientations = candidates[0].getHypotheses().size();

  // Create images in batches if required (less memory usage).
  if (create_image_batches_)
  {
    int batch_size = classifier_->getBatchSize();
    int num_iterations = (int) ceil(candidates.size() * num_orientations / (double) batch_size);
    int step_size = (int) floor(batch_size / (double) num_orientations);
    std::cout << " num_iterations: " << num_iterations << ", step_size: " << step_size << "\n";

    // Process the grasp candidates in batches.
    for (int i = 0; i < num_iterations; i++)
    {
      std::cout << i << "\n";
      std::vector<GraspSet>::iterator start = candidates.begin() + i * step_size;
      std::vector<GraspSet>::iterator stop;
      if (i < num_iterations - 1)
      {
        stop = candidates.begin() + i * step_size + step_size;
      }
      else
      {
        stop = candidates.end();
      }

      std::vector<GraspSet> hand_set_sublist(start, stop);
      std::vector<cv::Mat> image_list = learning_->createImages(cloud_cam, hand_set_sublist);

      std::vector<Grasp> valid_grasps;
      std::vector<cv::Mat> valid_images;
      extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);

      std::vector<float> scores_sublist = classifier_->classifyImages(valid_images);
      scores.insert(scores.end(), scores_sublist.begin(), scores_sublist.end());
      grasp_list.insert(grasp_list.end(), valid_grasps.begin(), valid_grasps.end());
    }
  }
  else
  {
    // Create the grasp images.
    std::vector<cv::Mat> image_list = learning_->createImages(cloud_cam, candidates);
    std::cout << " Image creation time: " << omp_get_wtime() - t0 << std::endl;

    std::vector<Grasp> valid_grasps;
    std::vector<cv::Mat> valid_images;
    extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);

    // Classify the grasp images.
    double t0_prediction = omp_get_wtime();
    scores = classifier_->classifyImages(valid_images);
    grasp_list.assign(valid_grasps.begin(), valid_grasps.end());
    std::cout << " Prediction time: " << omp_get_wtime() - t0 << std::endl;
  }

  // Select grasps with a score of at least <min_score_diff_>.
  std::vector<Grasp> valid_grasps;

  for (int i = 0; i < grasp_list.size(); i++)
  {
    if (scores[i] >= min_score_diff_)
    {
      std::cout << "grasp #" << i << ", score: " << scores[i] << "\n";
      valid_grasps.push_back(grasp_list[i]);
      valid_grasps[valid_grasps.size() - 1].setScore(scores[i]);
      valid_grasps[valid_grasps.size() - 1].setFullAntipodal(true);
    }
  }

  std::cout << "Found " << valid_grasps.size() << " grasps with a score >= " << min_score_diff_ << "\n";
  std::cout << "Total classification time: " << omp_get_wtime() - t0 << std::endl;

  if (plot_valid_grasps_)
  {
    Plot plotter;
    const HandSearch::Parameters& params = candidates_generator_->getHandSearchParams();
    plotter.plotFingers3D(valid_grasps, cloud_cam.getCloudOriginal(), "Valid Grasps", params.hand_outer_diameter_,
      params.finger_width_, params.hand_depth_, params.hand_height_);
  }

  return valid_grasps;
}

std::vector<GraspSet> GraspDetector::filterGraspsWorkspace(const std::vector<GraspSet>& hand_set_list,
  const std::vector<double>& workspace)
{
  int remaining = 0;
  std::vector<GraspSet> hand_set_list_out;

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid = hand_set_list[i].getIsValid();

    for (int j = 0; j < hands.size(); j++)
    {
      if (is_valid(j))
      {
        double half_width = 0.5 * outer_diameter_;
        Eigen::Vector3d left_bottom = hands[j].getGraspBottom() + half_width * hands[j].getBinormal();
        Eigen::Vector3d right_bottom = hands[j].getGraspBottom() - half_width * hands[j].getBinormal();
        Eigen::Vector3d left_top = hands[j].getGraspTop() + half_width * hands[j].getBinormal();
        Eigen::Vector3d right_top = hands[j].getGraspTop() - half_width * hands[j].getBinormal();
        Eigen::Vector3d approach = hands[j].getGraspBottom() - 0.05 * hands[j].getApproach();
        Eigen::VectorXd x(5), y(5), z(5);
        x << left_bottom(0), right_bottom(0), left_top(0), right_top(0), approach(0);
        y << left_bottom(1), right_bottom(1), left_top(1), right_top(1), approach(1);
        z << left_bottom(2), right_bottom(2), left_top(2), right_top(2), approach(2);
        double aperture = hands[j].getGraspWidth();

        if (aperture >= min_aperture_ && aperture <= max_aperture_ // make sure the object fits into the hand
          && x.minCoeff() >= workspace[0] && x.maxCoeff() <= workspace[1] // avoid grasping outside the x-workspace
          && y.minCoeff() >= workspace[2] && y.maxCoeff() <= workspace[3] // avoid grasping outside the y-workspace
          && z.minCoeff() >= workspace[4] && z.maxCoeff() <= workspace[5]) // avoid grasping outside the z-workspace
        {
          is_valid(j) = true;
          remaining++;
        }
        else
        {
          is_valid(j) = false;
        }
      }
    }

    if (is_valid.any())
    {
      hand_set_list_out.push_back(hand_set_list[i]);
      hand_set_list_out[hand_set_list_out.size() - 1].setIsValid(is_valid);
    }
  }

  ROS_INFO_STREAM("# grasps within workspace and gripper width: " << remaining);

  return hand_set_list_out;
}


std::vector<GraspSet> GraspDetector::filterHalfAntipodal(const std::vector<GraspSet>& hand_set_list)
{
  int remaining = 0;
  std::vector<GraspSet> hand_set_list_out;

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid = hand_set_list[i].getIsValid();

    for (int j = 0; j < hands.size(); j++)
    {
      if (is_valid(j))
      {
        if (!hands[j].isHalfAntipodal() || hands[j].isFullAntipodal())
        {
          is_valid(j) = true;
          remaining++;
        }
        else
        {
          is_valid(j) = false;
        }
      }
    }

    if (is_valid.any())
    {
      hand_set_list_out.push_back(hand_set_list[i]);
      hand_set_list_out[hand_set_list_out.size() - 1].setIsValid(is_valid);
    }
  }

  ROS_INFO_STREAM("# grasps that are not half-antipodal: " << remaining);

  return hand_set_list_out;
}


std::vector<Grasp> GraspDetector::extractHypotheses(const std::vector<GraspSet>& hand_set_list)
{
  std::vector<Grasp> hands_out;
  hands_out.resize(0);

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();

    for (int j = 0; j < hands.size(); j++)
    {
      if (hand_set_list[i].getIsValid()(j))
      {
        hands_out.push_back(hands[j]);
      }
    }
  }

  return hands_out;
}


void GraspDetector::extractGraspsAndImages(const std::vector<GraspSet>& hand_set_list,
  const std::vector<cv::Mat>& images, std::vector<Grasp>& grasps_out, std::vector<cv::Mat>& images_out)
{
  grasps_out.resize(0);
  images_out.resize(0);
  int num_orientations = hand_set_list[0].getHypotheses().size();

  for (int i = 0; i < hand_set_list.size(); i++)
  {
    const std::vector<Grasp>& hands = hand_set_list[i].getHypotheses();

    for (int j = 0; j < hands.size(); j++)
    {
      if (hand_set_list[i].getIsValid()(j))
      {
        grasps_out.push_back(hands[j]);
        images_out.push_back(images[i * num_orientations + j]);
      }
    }
  }
}

void GraspDetector::extractGraspsAndImages(const std::vector<Grasp>& hand_set_list,
  const std::vector<cv::Mat>& images, std::vector<Grasp>& grasps_out, std::vector<cv::Mat>& images_out)
{
  grasps_out.resize(0);
  images_out.resize(0);
  for (int i = 0; i < images.size(); i++)
  {
      grasps_out.push_back(hand_set_list[i]);
      images_out.push_back(images[i]);
  }
}

std::vector<Grasp> GraspDetector::findClusters(const std::vector<Grasp>& grasps)
{
  return clustering_->findClusters(grasps);
}

void GraspDetector::feature_vec(Grasp& elite_grasp, arma::mat& arma_v)
{
    Eigen::Vector3d v_bottom,v_surface,v_sample;
    Eigen::Matrix3d v_frame;
    Grasp vector_grasp=elite_grasp;
    v_bottom=vector_grasp.getGraspBottom();
    v_surface=vector_grasp.getGraspSurface();
    v_frame=vector_grasp.getFrame();
    v_sample=vector_grasp.getSample();
    Eigen::MatrixXd v(v_bottom.rows(),v_bottom.cols()+v_surface.cols()+v_frame.cols()+v_sample.cols());
    v << v_bottom, v_surface, v_frame, v_sample;
    ROS_INFO_STREAM("elite_grasp.pose.bottom_"<<v);
    arma_v = arma::mat(v.data(), v.rows(), v.cols(),false, false);
    arma_v.reshape(arma_v.n_rows*arma_v.n_cols,1);
    std::cout<< "arma_v:"<<arma_v<<std::endl;
    //return  arma_v;
}

std::vector<Grasp> GraspDetector::from_feature_vec( arma::mat& v)
{
  std::vector<Grasp> grasps;
  Eigen::MatrixXd eigen_v = Eigen::Map<Eigen::MatrixXd>(v.memptr(),
                                                        v.n_rows,
                                                        v.n_cols);
  ROS_INFO_STREAM("eigen_v"<<eigen_v.rows()<<"eigen_v cols :"<<eigen_v.cols());
  for (int i=0; i<v.n_cols;i++)
  {
    Eigen::MatrixXd grasp_mat;
    grasp_mat=eigen_v.block<18,1>(0,i);
    grasp_mat.resize(3,6);
    Grasp vector_grasp;
    vector_grasp.pose_.bottom_=grasp_mat.block<3,1>(0,0);
    vector_grasp.pose_.surface_=grasp_mat.block<3,1>(0,1);
    vector_grasp.pose_.frame_=grasp_mat.block<3,3>(0,2);
    vector_grasp.sample_=grasp_mat.block<3,1>(0,5);
   // ROS_INFO_STREAM("sample"<<vector_grasp.sample_);
    Eigen::Vector3d pos_;
    pos_<< 0.06,0.0,0.0;//0.06 is hand width
    vector_grasp.pose_.top_ = vector_grasp.pose_.bottom_+vector_grasp.pose_.frame_ * pos_;
    //ROS_INFO_STREAM("pose.top"<<vector_grasp.pose_.top_);
    Eigen::Vector3d pos_bottom;
    pos_bottom=vector_grasp.pose_.frame_.inverse()*(vector_grasp.pose_.bottom_-vector_grasp.sample_);
    vector_grasp.config_1d_.bottom_ = pos_bottom(0);
    vector_grasp.config_1d_.top_ = pos_bottom(0)+0.06;
    vector_grasp.config_1d_.center_ = pos_bottom(1);
   // ROS_INFO_STREAM("vector_grasp"<<vector_grasp.config_1d_.bottom_<<"top_"<<vector_grasp.config_1d_.top_ << "config_center"<< vector_grasp.config_1d_.center_<< "pos.bottom"<< vector_grasp.pose_.bottom_ << "pos.top"<< vector_grasp.pose_.top_ );
    grasps.push_back(vector_grasp);
  }
    return grasps;
}
