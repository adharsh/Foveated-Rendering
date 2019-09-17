#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <iostream>
#include <vector>

#include "Texture.h"

Texture::Texture(const std::string& filepath, bool interpolate)
	: filepath(filepath)
{
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	GLenum option = interpolate ? GL_LINEAR : GL_NEAREST;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, option);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, option);
	
	stbi_set_flip_vertically_on_load(true);
	data = stbi_load(filepath.c_str(), &width, &height, &num_channels, 0);
	if (data)
	{
		if (num_channels == 3)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		else if (num_channels == 4)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::load_data(const std::string& filepath, unsigned char** out_data, int* out_width, int* out_height, int* out_num_channels)
{
	stbi_set_flip_vertically_on_load(true);
	*out_data = stbi_load(filepath.c_str(), out_width, out_height, out_num_channels, 0);
}

void Texture::write_to_bmp(const std::string& filepath)
{
	stbi_flip_vertically_on_write(1); //non-zero value
	stbi_write_bmp(filepath.c_str(), width, height, num_channels, data);
}

Texture::Texture(unsigned char* data, int width, int height, int num_channels, bool interpolate)
{
	this->data = data;
	this->width = width;
	this->height = height;
	this->num_channels = num_channels;

	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	GLenum option = interpolate ? GL_LINEAR : GL_NEAREST;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, option);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, option);

	if (data)
	{
		if (num_channels == 3)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		else if (num_channels == 4)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "Passed in image data is null" << std::endl;
	}
	
	glBindTexture(GL_TEXTURE_2D, 0);
}

const std::vector<int> Texture::getPixelValue(unsigned int row, unsigned int col) const
{
	unsigned char* pixelOffset = data + (row * getWidth() +  col) * num_channels;
	unsigned char r = pixelOffset[0];
	unsigned char g = pixelOffset[1];
	unsigned char b = pixelOffset[2];

	if(num_channels >= 4)
		return {r, g, b, pixelOffset[3]};
	
	return {r, g, b};
};

Texture::~Texture()
{
	if(data)
		stbi_image_free(data);
	glDeleteTextures(1, &texture_id);
}

void Texture::deleteTexture() const
{
	glDeleteTextures(1, &texture_id);
}