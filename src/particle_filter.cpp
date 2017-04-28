/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

	num_particles = 7;
	default_random_engine gen;
	// Creating a normal (Gaussian) distribution for x, y, theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle particle;
		particle.id = i;
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_theta;
		particle.weight = 1.;
		particles.push_back(particle);
		// Print your samples to the terminal.
		cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " << sample_theta << endl;
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	//avoid division by zero
	default_random_engine gen;
	// Creating a normal (Gaussian) distribution for x, y, theta
	normal_distribution<double> dist_x(0., std_pos[0]);
	normal_distribution<double> dist_y(0., std_pos[1]);
	normal_distribution<double> dist_theta(0., std_pos[2]);

	for(int i = 0; i < particles.size(); i++){
		double x, y, theta, px, py, p_theta, noise_x, noise_y, noise_theta;
		x = particles[i].x;
		y = particles[i].y;
		theta = particles[i].theta;

		if (fabs(yaw_rate) > 0.001) {
			px = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			py = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			p_theta = theta+delta_t*yaw_rate;
		} else {
			px = x + velocity * delta_t * cos(theta);
			py = y + velocity * delta_t * sin(theta);
			p_theta = theta;
		}



		noise_x = dist_x(gen);
		noise_y = dist_y(gen);
		noise_theta = dist_theta(gen);

		particles[i].x = px + noise_x;
		particles[i].y = py + noise_y;
//		normalize angle
		double angle = p_theta + noise_theta;
		double norm = atan2(sin(angle),cos(angle));
		particles[i].theta = norm;

	}





}

vector<LandmarkObs>
ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<Map::single_landmark_s> landmarks, double range) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.

	for (int i = 0; i < predicted.size(); i++){
		double x, y, nearest;
		int nearest_id;
		x = predicted[i].x;
		y = predicted[i].y;
		nearest = 0;
		for (int l = 0; l < landmarks.size(); l++){
			double l_x = landmarks[l].x_f;
			double l_y = landmarks[l].y_f;
			double temp_d = sqrt((x-l_x)*(x-l_x)+ (y-l_y)*(y-l_y));
			if (temp_d > range){continue;}
			if (nearest == 0){
				nearest = temp_d;
				nearest_id = landmarks[l].id_i;
			}
			else if(nearest > temp_d){
				nearest = temp_d;
				nearest_id = landmarks[l].id_i;
			}
		}
		predicted[i].id = nearest_id;
	}
	return predicted;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution


	double sum_w = 0;
	for(int i = 0; i < particles.size(); i++){
		double p_x, p_y, p_theta;
		p_x = particles[i].x;
		p_y = particles[i].y;
		p_theta = particles[i].theta;
		vector<LandmarkObs> predicted;
		for(int l = 0; l < observations.size(); l++){
			double local_x, local_y;
			LandmarkObs measured;
			local_x = observations[l].x;
			local_y = observations[l].y;
			measured.x = local_x*cos(p_theta) - local_y*sin(p_theta) + p_x;
			measured.y = local_x*sin(p_theta) + local_y*cos(p_theta) + p_y;
			predicted.push_back(measured);
		}

		vector<LandmarkObs> associated = dataAssociation(predicted, map_landmarks.landmark_list, sensor_range);
		double w = 0;
		for (int j = 0; j < associated.size(); j++){
			double x, y, temp_w;
//			landmark index begins form 1
			int i_land = associated[j].id;
//			check i_land = map_landmarks.landmark_list[i_land].id_i
			int check_id = map_landmarks.landmark_list[i_land-1].id_i;
			x = associated[j].x;
			y = associated[j].y;
			double l_x = map_landmarks.landmark_list[i_land-1].x_f;
			double l_y = map_landmarks.landmark_list[i_land-1].y_f;
			double range = sqrt((p_x-l_x)*(p_x-l_x)+ (p_y-l_y)*(p_y-l_y));
			if(range > sensor_range){continue;}
			double exp1 = (x-l_x)*(x-l_x)/(2*std_landmark[0]*std_landmark[0]);
			double exp2 = (y-l_y)*(y-l_y)/(2*std_landmark[1]*std_landmark[1]);
			temp_w = exp(-1*(exp1+exp2))/(2*M_PI*std_landmark[0]*std_landmark[1]);
			if (w == 0){w = temp_w;}
			else {w *= temp_w;}

		}
		particles[i].weight = w;
		sum_w += w;
	}

//	normalize weights
	for (int p = 0; p < particles.size(); p++){
		particles[p].weight /= sum_w;
	}

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.

	random_device seed_gen;
	mt19937 engine(seed_gen());

	vector<double> prob;
	for (int i = 0; i < particles.size(); i++) {
		prob.push_back(particles[i].weight);
	}

	discrete_distribution<size_t> dist(
			prob.begin(),
			prob.end()
	);

	for (size_t n = 0; n < particles.size(); ++n) {
		size_t result = dist(engine);
		particles[n] = particles[result];
	}
	}


void ParticleFilter::write(std::string filename) {
	std::ofstream dataFile;
  dataFile.open(filename, std::ofstream::out | std::ofstream::trunc);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << " "<< particles[i].weight <<"\n";
	}
	dataFile.close();
}
