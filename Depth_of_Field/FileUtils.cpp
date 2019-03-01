#include "FileUtils.h"

#include <fstream>
#include <sstream>

std::string FileUtils::read_file(const std::string& file_path)
{
	std::ifstream t(file_path);
	std::stringstream buffer;
	buffer << t.rdbuf();

	return buffer.str();
	/*
	std::ifstream t(file_path);
	t.seekg(0, std::ios::end);
	size_t size = t.tellg();
	std::string buffer(size, ' ');
	t.seekg(0);
	t.read(&buffer[0], size);
	*/
}