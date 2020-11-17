#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv_;

uniform sampler2D SAT;

out vec2 uv;
out vec2 pos;

void main()
{
	uv = uv_;
	gl_Position = vec4(position, 1.0);
	pos = position.xy;
}