/*  Copyright &copy; 2016, Stefan Sicklinger, Munich
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
#include "VisualizerSetting.h"
#include "SignalDataVisualizer.h"
#include "OutputDatabase.h"

VisualizerSetting::VisualizerSetting()
{
	// Initialize default property
	PROPERTY_SCALARBAR_TITLE = "";
	PROPERTY_SCALARBAR_VISIBILITY = true;		// Scalar Bar is enabled by Default
	PROPERTY_FRAMEID = 0;						// Set the initial frame as Zero
	PROPERTY_FIELD_TYPE = STACCATO_x_Re;
	PROPERTY_SCALING_FACTOR = 1;

	PROPERTY_ANIMATION_DURATION = 5;
	PROPERTY_ANIMATION_REPEAT   = 1;

	PROPERTY_RESULTS_AVALABLE = false;

	myViewModeLabelMap["Surface"] = STACCATO_FieldProperty_Surface;
	myViewModeLabelMap["Surface with Edges"] = STACCATO_FieldProperty_SurfaceWithEdges;
	myViewModeLabelMap["Wireframe"] = STACCATO_FieldProperty_Wireframe;
}

VisualizerSetting::~VisualizerSetting()
{
}

void VisualizerSetting::setCommuniationToFieldDataVisualizer(FieldDataVisualizer& _fieldDataVisualizer) {
	myFieldDataVisualizer = &_fieldDataVisualizer;
	myFieldDataVisualizer->connectVisualizerSetting(this);
	std::cout << ">> Connection to FieldDataVisualizer is set by VisualizerSetting.\n";
}

void VisualizerSetting::setCommuniationToSignalDataVisualizer(SignalDataVisualizer& _signalDataVisualizer) {
	mySignalDataVisualizer = &_signalDataVisualizer;
	mySignalDataVisualizer->connectVisualizerSetting(this);
	std::cout << ">> Connection to SignalDataVisualizer is set by VisualizerSetting.\n";
}

void VisualizerSetting::commitViewSetting(STACCATO_FieldProperty_type _property) {

	switch (_property)
	{
	case STACCATO_FieldProperty_Surface:
		myFieldDataSetting = new SurfaceSetting();
		std::cout << ">> Visualizer Setting set as: Surface.\n";
		break;
	case STACCATO_FieldProperty_SurfaceWithEdges:
		myFieldDataSetting = new SurfaceWithEdgesSetting();
		std::cout << ">> Visualizer Setting set as: SurfaceWithEdges.\n";
		break;
	case STACCATO_FieldProperty_Wireframe:
		myFieldDataSetting = new WireFrameSetting();
		std::cout << ">> Visualizer Setting set as: Wireframe.\n";
		break;
	default:
		std::cerr << "Unidentified VisualizerSetting.\n";
		break;
	}
	
	updateSetting();
}

void VisualizerSetting::commitVectorFieldComponent(STACCATO_VectorField_components _component) {
	PROPERTY_FIELD_TYPE = _component;
}

void VisualizerSetting::updateSetting() {
	stopAnimation();			// If animation is played
	myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetScalar(PROPERTY_FIELD_TYPE, PROPERTY_FRAMEID);
	myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetVector(PROPERTY_FRAMEID);
	myFieldDataVisualizer->plotVectorField();																	// Update Plot
}

void VisualizerSetting::commitScalarBar(bool _enable) {
	PROPERTY_SCALARBAR_VISIBILITY = _enable;
}

void VisualizerSetting::commitScalingFactor(double _scalingFactor) {
	PROPERTY_SCALING_FACTOR = _scalingFactor;
}

void VisualizerSetting::commitCurrentFrame(int _frameID) {
	std::cout << ">> Committed to Frame: " << _frameID << std::endl;
	PROPERTY_FRAMEID = _frameID;
}

void VisualizerSetting::setScalarbarTitle(std::string _title) {
	PROPERTY_SCALARBAR_TITLE = new char[_title.size() + 1];
	strcpy(PROPERTY_SCALARBAR_TITLE, _title.c_str());
}

void VisualizerSetting::generateCaseAnimation(std::vector<int> &_frameIndices) {

	myFieldDataVisualizer->animate(PROPERTY_FIELD_TYPE, _frameIndices, false);
	myFieldDataVisualizer->plotVectorFieldAtIndex(0);
}

void VisualizerSetting::generateHarmonicAnimation(std::vector<int> &_frameIndices) {

	myFieldDataVisualizer->animate(PROPERTY_FIELD_TYPE, _frameIndices, true);
	myFieldDataVisualizer->plotVectorFieldAtIndex(0);
}

void VisualizerSetting::playAnimation() {
	myFieldDataVisualizer->myAnimationScenePlayProc(PROPERTY_ANIMATION_DURATION, PROPERTY_ANIMATION_REPEAT);
}

void VisualizerSetting::stopAnimation() {
	myFieldDataVisualizer->myAnimationSceneStopProc();
}

void VisualizerSetting::visualizeAnimationFrames(int _duration, int _repeat) {
	PROPERTY_ANIMATION_DURATION = _duration;
	PROPERTY_ANIMATION_REPEAT = _repeat;

	myFieldDataVisualizer->setViewMode(true);
	playAnimation();
	myFieldDataVisualizer->setViewMode(false);
}

void VisualizerSetting::setResultAvailable(bool _available) {
	PROPERTY_RESULTS_AVALABLE = _available;
}

void VisualizerSetting::setCurrentAnalysis(std::string _analysisName) {
	PROPERTY_CURRENT_ANALYSIS_INDEX = myFieldDataVisualizer->getHMesh()->myOutputDatabase->findAnalysis(_analysisName);
	PROPERTY_CURRENT_TIMESTEP_INDEX = 0;
	PROPERTY_CURRENT_LOADCASE_INDEX = 0;

	commitCurrentFrame(myFieldDataVisualizer->getHMesh()->myOutputDatabase->myAnalyses[PROPERTY_CURRENT_ANALYSIS_INDEX].startIndex);						// Commit the frame as analysis
}

void VisualizerSetting::updateCurrentTimeStepIndex(int _analysisIndex) {
	PROPERTY_CURRENT_TIMESTEP_INDEX = myFieldDataVisualizer->getHMesh()->myOutputDatabase->myAnalyses[_analysisIndex].startIndex;
}

void VisualizerSetting::updateCurrentLoadCaseIndex(int _timeStepIndex) {
	PROPERTY_CURRENT_LOADCASE_INDEX = myFieldDataVisualizer->getHMesh()->myOutputDatabase->myAnalyses[PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[_timeStepIndex].startIndex;
}

void VisualizerSetting::commitTimeStepIndex(int _timeStepIndex) {
	PROPERTY_CURRENT_TIMESTEP_INDEX = _timeStepIndex;
	PROPERTY_CURRENT_LOADCASE_INDEX = 0;

	commitCurrentFrame(myFieldDataVisualizer->getHMesh()->myOutputDatabase->myAnalyses[PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[PROPERTY_CURRENT_TIMESTEP_INDEX].startIndex);						// Commit the frame as timestep
}

void VisualizerSetting::commitLoadCaseIndex(int _loadCaseIndex) {
	PROPERTY_CURRENT_LOADCASE_INDEX = _loadCaseIndex;

	commitCurrentFrame(myFieldDataVisualizer->getHMesh()->myOutputDatabase->myAnalyses[PROPERTY_CURRENT_ANALYSIS_INDEX].timeSteps[PROPERTY_CURRENT_TIMESTEP_INDEX].caseList[PROPERTY_CURRENT_LOADCASE_INDEX].startIndex);						// Commit the frame as loadCase
}

void VisualizerSetting::listProperties() {
	std::cout << "========= Visualizer Settings =========" << std::endl;
	std::cout << "PROPERTY_CURRENT_ANALYSIS_INDEX: " << PROPERTY_CURRENT_ANALYSIS_INDEX << std::endl;
	std::cout << "PROPERTY_CURRENT_TIMESTEP_INDEX: " << PROPERTY_CURRENT_TIMESTEP_INDEX << std::endl;
	std::cout << "PROPERTY_CURRENT_LOADCASE_INDEX: " << PROPERTY_CURRENT_LOADCASE_INDEX << std::endl;
	std::cout << "=======================================" << std::endl;
}

void VisualizerSetting::commitToNextTimeStep() {
	if (PROPERTY_CURRENT_TIMESTEP_INDEX + 1 < myFieldDataVisualizer->getHMesh()->myOutputDatabase->getNumberOfTimeSteps(PROPERTY_CURRENT_ANALYSIS_INDEX)) {
		commitTimeStepIndex(PROPERTY_CURRENT_TIMESTEP_INDEX+1);
		std::cout << ">> Commited to Next Time Step.\n";
	}
	else
		std::cout << ">> Cannot Commit Further.\n";
}

void VisualizerSetting::commitToPerviousTimeStep() {
	if (PROPERTY_CURRENT_TIMESTEP_INDEX > 0) {
		commitTimeStepIndex(PROPERTY_CURRENT_TIMESTEP_INDEX-1);
		std::cout << ">> Commited to Pervious Time Step.\n";
	}
	else
		std::cout << ">> Cannot Commit Further.\n";
}