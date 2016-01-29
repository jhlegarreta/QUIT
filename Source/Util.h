/*
 *  Util.h
 *  Part of the QUantitative Image Toolbox
 *
 *  Copyright (c) 2015 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#ifndef QUIT_UTIL_H
#define QUIT_UTIL_H

#include <string>
#include <map>
#include <vector>
#include <random>
#include <functional>
#include <mutex>
#include <locale>
#include <vector>

#include <Eigen/Dense>

#include "itkCommand.h"
#include "itkVectorMagnitudeImageFilter.h"

#include "Types.h"

namespace QI {

const std::string &OutExt(); //!< Return the extension stored in $QUIT_EXT
std::string StripExt(const std::string &filename);
std::mt19937_64::result_type RandomSeed(); // Thread-safe random seed

void writeResiduals(const typename VectorImageF::Pointer img, const std::string prefix, const bool allResids = false);

template<typename TImg = QI::ImageF>
auto ReadImage(const std::string &fname) -> typename TImg::Pointer {
    typedef itk::ImageFileReader<TImg> TReader;
    typename TReader::Pointer file = TReader::New();
    file->SetFileName(fname);
    file->Update();
    typename TImg::Pointer img = file->GetOutput();
    img->DisconnectPipeline();
    return img;
}

template<typename TImg = QI::ImageF>
void writeResult(const typename TImg::Pointer img, const std::string path) {
    typedef itk::ImageFileWriter<TImg> TWriter;
    auto file = TWriter::New();
    file->SetFileName(path);
    file->SetInput(img);
    file->Update();
}

class GenericMonitor : public itk::Command {
public:
	typedef GenericMonitor            Self;
	typedef itk::Command            Superclass;
	typedef itk::SmartPointer<Self> Pointer;
	itkTypeMacro(GenericMonitor, Superclass);
	itkNewMacro(Self);
protected:
	GenericMonitor() {}
public:
    void Execute(itk::Object *caller, const itk::EventObject &event) override {
        Execute((const itk::Object *)caller, event);
    }
    void Execute(const itk::Object *object, const itk::EventObject &event) override {
		const itk::ProcessObject *filter = static_cast<const itk::ProcessObject *>(object);
		if (typeid(event) == typeid(itk::ProgressEvent)) {
			std::cout << "Progress: " << (filter->GetProgress()*100) << "% complete" << std::endl;
		} else {
			std::cout << "Received event: " << typeid(event).name() << std::endl;
		}
	}
};

template <typename TFilter>
class SliceMonitor : public itk::Command {
public:
	typedef SliceMonitor            Self;
	typedef itk::Command            Superclass;
	typedef itk::SmartPointer<Self> Pointer;
	itkTypeMacro(SliceMonitor, Superclass);
	itkNewMacro(Self);
protected:
	SliceMonitor() {}
public:
    void Execute(itk::Object *caller, const itk::EventObject &event) override {
		Execute((const itk::Object *)caller, event);
	}
    void Execute(const itk::Object *object, const itk::EventObject & event) override {
		const TFilter *filter = static_cast<const TFilter *>(object);
		if (typeid(event) == typeid(itk::ProgressEvent)) {
			std::cout << "Completed slice " << filter->GetSliceIndex() << std::endl;
		} else if (typeid(event) == typeid(itk::IterationEvent)) {
			std::cout << "Starting slice " << filter->GetSliceIndex() << std::endl;
		} else {
			std::cout << "Received event: " << typeid(event).name() << std::endl;
		}
	}
};

/*
 *  The following is an interesting way to get rid of [] & commas in a stream. Allows ITK vectors to be extracted directly
 *  From http://stackoverflow.com/questions/1894886/parsing-a-comma-delimited-stdstring
 */
class CSVHack : public std::ctype<char> {
public:
    CSVHack() : std::ctype<char>(get_table()) {}

    static std::ctype_base::mask const* get_table() {
    	//table_size is a static member of ctype<char>
        static std::vector<std::ctype_base::mask> rc(table_size, std::ctype_base::mask());

        rc[','] = std::ctype_base::space;
        rc['\n'] = std::ctype_base::space;
        rc[' '] = std::ctype_base::space;
        rc['['] = std::ctype_base::space;
		rc[']'] = std::ctype_base::space;
        return &rc[0];
    }
}; 

template<typename T> bool Read(const std::string &s, T &val) {
	std::istringstream stream(s);
	if (!(stream >> val)) {
		throw(std::runtime_error("Failed to parse input: " + s));
	}
	return true;
}

template<typename T> bool Read(std::istream &in, T &val) {
	std::string line;
	// Ignore comment lines. Use shell script convention
	while (in.peek() == '#') {
		if (!std::getline(in, line))
			throw(std::runtime_error("Failed to read input."));
	}
	if (!std::getline(in, line)) {
		throw(std::runtime_error("Failed to read input. Last line was: " + line));
	}
	return Read(line, val);
}

template<typename Scalar>
void ReadArray(const std::string &s, Eigen::Array<Scalar, Eigen::Dynamic, 1> &array) {
	std::istringstream stream(s);
	std::vector<Scalar> vals;

	Scalar temp;
	while (stream >> temp) {
		vals.push_back(temp);
	}

	array = Eigen::Array<Scalar, Eigen::Dynamic, 1>(vals.size());
	for (int i = 0; i < vals.size(); i++) {
		array[i] = vals[i];
	}
}

template<typename Scalar>
void ReadArray(std::istream &in, Eigen::Array<Scalar, Eigen::Dynamic, 1> &array) {
	std::string line;
	// Ignore comment lines. Use shell script convention
	while (in.peek() == '#') {
		if (!std::getline(in, line))
			throw(std::runtime_error("Failed to read input."));
	}
	if (!std::getline(in, line)) {
		throw(std::runtime_error("Failed to read input."));
	}
	ReadArray(line, array);
}

} // End namespace QUIT

#endif // QUIT_UTIL_H
