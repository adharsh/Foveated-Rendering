#pragma once

#include <glad/glad.h>
#include <string>

class Texture //doesn't support texture blending
{
private:
	GLuint texture_id;
	int width, height;
	int num_channels; //3 for rgb, 4 for rgba
public:
	Texture(const std::string& filepath, bool interpolate = true);
	~Texture();
	void deleteTexture() const;
	
	void bind() const { glBindTexture(GL_TEXTURE_2D, texture_id); }
	void unbind() const { glBindTexture(GL_TEXTURE_2D, 0); }

	const unsigned int getWidth() const { return width; }
	const unsigned int getHeight() const { return height; }
	const unsigned int getNumChannels() const { return num_channels; }
	const GLuint getID() const { return texture_id; }
};