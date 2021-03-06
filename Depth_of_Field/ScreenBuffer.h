#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include "Shader.h"


class ScreenBuffer : public Shader
{
private:
	GLuint FBO; //frame buffer object
	GLuint RBO; //render buffer object
	GLuint textureID; //texture color buffer
	GLuint quadVAO; //rectangle
	GLuint quadVBO; //rectangle
	glm::vec4 clear_color;
public:
	ScreenBuffer(const std::string& vertex_shader_path, const std::string& fragment_shader_path, 
		unsigned int width, unsigned int height, glm::vec4 clear_color, bool depth, bool stencil);
	~ScreenBuffer();

	void drawToTexture() const;
	void drawTextureToScreen() const;
	static void initalize(const glm::vec4& clearColor = glm::vec4());

	void bindBuffer() const;
	static void bindDefaultBuffer();

	static void clearColor(const glm::vec4& clear_color);
	static void clearBuffer(bool colorBuffer, bool depthBuffer, bool stencilBuffer);

	static void enableDepthTest() { glEnable(GL_DEPTH_TEST); }
	static void enableStencilTest() { glEnable(GL_STENCIL_TEST); }
	static void disableDepthTest() { glDisable(GL_DEPTH_TEST); }
	static void disableStencilTest() { glDisable(GL_STENCIL_TEST); }

	static void drawAsWireframe() { glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); }
	static void drawAsPoints() { glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); }
	static void drawAsFilled() { glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); }
};