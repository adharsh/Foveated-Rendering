#include "Shader.h"
#include "Window.h"
#include "Texture.h"
#include "cuda/deinterleave.cuh"
#include "cuda/summed_area_table.cuh"

#include <algorithm>
#include <fstream>
#include <iostream>


//ASSUMPTIONS: 
//input is square image
//Minimum max resolution is 2048 resolution (gaurenteed to work up to at most 2048x2048 image
	//https://stackoverflow.com/questions/7954927/passing-a-list-of-values-to-fragment-shader
//MAX_BOXES = 10
//other shader constants


#define gen 0

int main()
{
#if !gen
	int width, height, num_channels;
	width = height = 256; num_channels = 3;
	int scale = 5;
	Window win = Window("Depth of Field", scale * width, scale * height, glm::vec4(0, 1, 1, 0));

	Shader shader = Shader();
	shader.addVertexShader("res/RectVS.c");
	shader.addFragmentShader("res/RectFS.c");
	shader.compileShader();
	
	std::vector<float> frame(3 * width * height);
	FILE* fp;
	fp = fopen("res/lena_256.planar", "rb");
	fread(&frame[0], sizeof(float), 3 * width * height, fp);
	fclose(fp);
	
	/*std::ofstream f;
	f.open("res/sat_frame_test.csv");
	int c = 0;
	for (int c = 0; c < width; c++)
	{
		for (int ro = 0; ro < height; ro++)
		{
			int i = width * ro + c;
			f << "(" << frame[i] << ":" << frame[i + width*height] << ":" << frame[i + 2 * width * height] << "),";
		}
		f << "\n";
	}	
	f.close();*/

#else
	int width, height, num_channels;
	unsigned char* img = NULL;
	Texture::load_data("res/lena_256.png", &img, &width, &height, &num_channels);
	int scale = 5;
	Window win = Window("Depth of Field", scale * width, scale * height, glm::vec4(0, 1, 1, 0));

	unsigned int N = width;
	//N is dynamic so malloc is needed (or vector)
	float* r = (float*)malloc(N * N * sizeof(float));
	float* g = (float*)malloc(N * N * sizeof(float));
	float* b = (float*)malloc(N * N * sizeof(float));

	deinterleave(img, N * N, &r, &g, &b);
	summed_area_table(r, N);
	summed_area_table(g, N);
	summed_area_table(b, N);

	FILE* fp;
	fp = fopen("res/lena_256.planar", "wb");
	fwrite(r, sizeof(float), width * height, fp);
	fwrite(g, sizeof(float), width * height, fp);
	fwrite(b, sizeof(float), width * height, fp);
	fclose(fp);
	
	/*std::ofstream f;
	f.open("res/sat_frame.csv");
	int c = 0;
	for (int c = 0; c < width; c++)
	{
		for (int ro = 0; ro < height; ro++)
		{
			int i = width * ro + c;
			f << "(" << r[i] << ":" << g[i] << ":" << b[i] << "),";
		}
		f << "\n";
	}	
	f.close();*/

	free(r);
	free(g);
	free(b);
	
	system("pause");

	std::vector<float> frame(3 * width * height);
	for (int i = 0; i < width * height; i++)
		frame[i] = r[i];

	for (int i = 0; i < width * height; i++)
		frame[i + width * height] = g[i];

	for (int i = 0; i < width * height; i++)
		frame[i + 2 * width * height] = b[i];

	Shader shader = Shader();
	shader.addVertexShader("res/RectVS.c");
	shader.addFragmentShader("res/RectFS.c");
	shader.compileShader();
#endif

	//id - 1 -> lower-left rect coord val
	//-1 -> can't use, the same val
	//0 -> 3, can't use (base case)
	//1 -> 6, box_dim = (array[1] - array[0])*3
	//etc.
	//int num_boxes = (int)(std::log(std::max(width, height)) / std::log(3));
	int num_boxes = 10;
	int* box_coords = new int[num_boxes];
	box_coords[0] = 0;
	box_coords[1] = 2;
	std::cout << "0 2 ";
	for (unsigned int i = 2; i < num_boxes; i++)
	{
		box_coords[i] = box_coords[i - 1] + std::pow(3, i - 1);
		std::cout << box_coords[i] << " ";
	}
	std::cout << std::endl;
	
	std::vector<GLfloat> data = 
	{
		//Positions		UVs
		1, 1, 0,		1, 1, //top right
		1, -1, 0,		1, 0, // bottom right
		-1, -1, 0,		0, 0, // bottom left
		-1,  1, 0,		0, 1  // top left
	};

	std::vector<GLuint> indices = 
	{
		0, 1, 3,  // first triangle
		1, 2, 3   // second triangle
	};

	unsigned int VAO, VBO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	
	glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(GLfloat), &data[0], GL_STATIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*) 0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*) (3 * sizeof(GLfloat)));
	
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	//box_coords = new int[num_boxes];
	//only have to do once, not too slow since only once
	shader.bind();
	shader.setUniform1iv("box_coords", box_coords, num_boxes);
	shader.setUniform1f("width", width); //don't worry about scaling, it's gonna end up as [0,1] anyways
	shader.setUniform1f("height", height);

	//TODO: make this oop and encapsulated
	GLuint ssbo;
	glGenBuffers(1, &ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, 3 * width * height * sizeof(float), &frame[0], GL_STATIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	shader.unbind();

	while (!win.closed())
	{
		win.clear();

		shader.bind();
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

		double x, y;
		win.getMousePosition(x, y);
		//std::cout << "(" << x << ", " << y << ")" << "\n";

		float x_transformed = (x / ((double)win.getWidth()))*2.0f - 1.0f;
		float y_transformed = ((((double)win.getHeight()) - y) / ((double)win.getHeight()))*2.0f - 1.0f;
		//TODO, directly convert to UV coordinates
		shader.setUniform2f("eye_pos", glm::vec2(x_transformed, y_transformed));

		//glActiveTexture(GL_TEXTURE0);
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		shader.unbind();

		win.update();
	}

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);

	return 0;
}