#ifndef SUBJECT_H
#define SUBJECT_H

#include "observer.h"
#include <vector>

class Subject {
public:

	virtual void attachObserver(Observer* observer) {
		this->observers.push_back(observer);
	}

	virtual void notify() {
		for (std::vector<Observer*>::iterator it = observers.begin();
			it != observers.end();
			++it) {
			(*it)->update();
		}
	}

	// attribute
protected:
	std::vector<Observer*> observers;
};

#endif // SUBJECT_H
