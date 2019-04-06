#pragma once

#include <glad/glad.h>
#include <string>

//enum class Color {RED, GREEN, BLUE, ALPHA};

class Texture //doesn't support texture blending
{
private:
	GLuint texture_id;
	int width, height;
	int num_channels; //3 for rgb, 4 for rgba
	unsigned char* data;
	std::string filepath;
public:
	Texture(const std::string& filepath, bool interpolate = true);
	Texture(const unsigned char* data, int width, int height, int num_channels, bool interpolate = true);
	~Texture();
	void deleteTexture() const;

	static void load_data(const std::string& filepath, unsigned char** out_data, int* out_width, int* out_height, int* out_num_channels);
	
	void bind() const { glBindTexture(GL_TEXTURE_2D, texture_id); }
	void unbind() const { glBindTexture(GL_TEXTURE_2D, 0); }

	const std::vector<int> getPixelValue(unsigned int row, unsigned int col) const;
	const unsigned int getWidth() const { return width; }
	const unsigned int getHeight() const { return height; }
	const unsigned int getNumChannels() const { return num_channels; }
	const GLuint getID() const { return texture_id; }
	const std::string getFilePath() const { return filepath; }
};