/*  Copyright &copy; 2017, Stefan Sicklinger, Munich
*
*  All rights reserved.
*
*  This file is part of STACCATO.
*
*  STACCATO is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  STACCATO is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with STACCATO.  If not, see http://www.gnu.org/licenses/.
*/
/*************************************************************************************************
* \file VisualizerSetting.h
* This file holds the class of VisualizerSetting.
* \date 3/2/2018
**************************************************************************************************/
#pragma once

#include <FieldDataVisualizer.h>
#include <FieldDataSetting.h>

#include <SurfaceSetting.h>
#include <SurfaceWithEdgesSetting.h>
#include <WireframeSetting.h>

using namespace STACCATO_Visualizer;
using namespace STACCATO_Results;

class SignalDataVisualizer;

class VisualizerSetting
{
public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Harikrishnan Sreekumar
	***********/
	VisualizerSetting();
	/***********************************************************************************************
	* \brief Destructor
	* \author Harikrishnan Sreekumar
	***********/
	~VisualizerSetting();
	/***********************************************************************************************
	* \brief Set the Field Data visualizer for communication
	* \param[in] _fieldDataVisualizer Field Data visualizer
	* \author Harikrishnan Sreekumar
	***********/
	void setCommuniationToFieldDataVisualizer(FieldDataVisualizer& _fieldDataVisualizer);
	/***********************************************************************************************
	* \brief Set the Signal Data visualizer for communication
	* \param[in] _signalDataVisualizer Signal Data visualizer
	* \author Harikrishnan Sreekumar
	***********/
	void setCommuniationToSignalDataVisualizer(SignalDataVisualizer& _signalDataVisualizer);
	/***********************************************************************************************
	* \brief Set the Visualization Type
	* \param[in] _property VisualizerProperty
	* \author Harikrishnan Sreekumar
	***********/
	void commitViewSetting(STACCATO_FieldProperty_type _property);
	/***********************************************************************************************
	* \brief Enable/Disable ScalarBar
	* \param[in] _enable Boolean
	* \author Harikrishnan Sreekumar
	***********/
	void commitScalarBar(bool _enable);
	/***********************************************************************************************
	* \brief Update the FieldDataVisualizer with Settings
	* \author Harikrishnan Sreekumar
	***********/
	void updateSetting();
	/***********************************************************************************************
	* \brief Set the component of vector result to be visualizer
	* \param[in] _component VectorComponent
	* \author Harikrishnan Sreekumar
	***********/
	void commitVectorFieldComponent(STACCATO_VectorField_components _component);
	/***********************************************************************************************
	* \brief Visualize the TimeFrame
	* \param[in] _frameId TimeFrameId
	* \author Harikrishnan Sreekumar
	***********/
	void commitCurrentFrame(int _frameId);
	/***********************************************************************************************
	* \brief Set ScalarBar Title
	* \param[in] _title Title Text
	* \author Harikrishnan Sreekumar
	***********/
	void setScalarbarTitle(std::string _title);
	/***********************************************************************************************
	* \brief Set Scaling Factor
	* \param[in] _scalingFactor
	* \author Harikrishnan Sreekumar
	***********/
	void commitScalingFactor(double _scalingFactor);
	/***********************************************************************************************
	* \brief Generate Frames for Animation
	* \author Harikrishnan Sreekumar
	***********/
	void generateSubCaseAnimation(std::vector<int> &_frameIndices);
	/***********************************************************************************************
	* \brief Generate Frames for Harmonic Animation
	* \author Harikrishnan Sreekumar
	***********/
	void generateHarmonicAnimation(std::vector<int> &_frameIndices);
	/***********************************************************************************************
	* \brief Set Animation Duration and Enable/Disable Repeat
	* \param[in] _duration
	* \param[in] _repeat
	* \author Harikrishnan Sreekumar
	***********/
	void visualizeAnimationFrames(int _duration, int _repeat);
	/***********************************************************************************************
	* \brief Play Animation
	* \author Harikrishnan Sreekumar
	***********/
	void playAnimation();
	/***********************************************************************************************
	* \brief Stop Animation
	* \author Harikrishnan Sreekumar
	***********/
	void stopAnimation();
	/***********************************************************************************************
	* \brief Stop Animation
	* \author Harikrishnan Sreekumar
	***********/
	void setResultAvailable(bool);
	/***********************************************************************************************
	* \brief Set Analysis for Visualization
	* \author Harikrishnan Sreekumar
	***********/
	void setCurrentAnalysis(std::string _analysisName);
	/***********************************************************************************************
	* \brief Update TimeStep for New Anaylsis
	* \author Harikrishnan Sreekumar
	***********/
	void updateCurrentTimeStepIndex(int _analysisIndex);
	/***********************************************************************************************
	* \brief Update LoadCase for New TimeStep
	* \author Harikrishnan Sreekumar
	***********/
	void updateCurrentLoadCaseIndex(int _timeStepIndex);
	/***********************************************************************************************
	* \brief Set TimeStep for Visualization
	* \author Harikrishnan Sreekumar
	***********/
	void commitTimeStepIndex(int _timeStepIndex);
	/***********************************************************************************************
	* \brief Set LoadCase for Visualization
	* \author Harikrishnan Sreekumar
	***********/
	void commitLoadCaseIndex(int _loadCaseIndex);
	/***********************************************************************************************
	* \brief List Properties
	* \author Harikrishnan Sreekumar
	***********/
	void listProperties();
	/***********************************************************************************************
	* \brief Update to next time step of the set analysis
	* \author Harikrishnan Sreekumar
	***********/
	void commitToNextTimeStep();
	/***********************************************************************************************
	* \brief Update to pervious time step of the set analysis
	* \author Harikrishnan Sreekumar
	***********/
	void commitToPerviousTimeStep();

private:
	// FieldDataVisualizer
	FieldDataVisualizer* myFieldDataVisualizer;
	// SignalDataVisualizer
	SignalDataVisualizer* mySignalDataVisualizer;
	// Chosen Property for Surface or SurfaceWithEdges or Wireframe
	STACCATO_FieldProperty_type myFieldProperty;

public:
	// Setting object for type: Surface or SurfaceWithEdges or Wireframe
	FieldDataSetting* myFieldDataSetting;
	// Current FrameId
	int PROPERTY_FRAMEID;
	// Current ScalingFactor
	double PROPERTY_SCALING_FACTOR;
	// Current Scalarbar Visibility
	bool PROPERTY_SCALARBAR_VISIBILITY;
	// Current Scalarbar Title
	char* PROPERTY_SCALARBAR_TITLE;
	// Current VectorField Component
	STACCATO_VectorField_components PROPERTY_FIELD_TYPE;
	// Animation Duration
	int PROPERTY_ANIMATION_DURATION;
	// Animation : Repeat/Once
	int PROPERTY_ANIMATION_REPEAT;
	// View Label to Enum Map
	std::map<std::string, STACCATO_FieldProperty_type> myViewModeLabelMap;
	// FeAnalysis performed
	bool PROPERTY_RESULTS_AVALABLE;
	// Current Analysis
	int PROPERTY_CURRENT_ANALYSIS_INDEX;
	// Current TimeStep
	int PROPERTY_CURRENT_TIMESTEP_INDEX;
	// Current Load Case
	int PROPERTY_CURRENT_LOADCASE_INDEX;
};
