#include "Shader.h"
#include "Window.h"
#include "Texture.h"
#include "cuda/deinterleave.cuh"
#include "cuda/summed_area_table.cuh"

#include <algorithm>
#include <fstream>
#include <iostream>

//	0 2 5 14 41

int main()
{
	int dim = 256;
	int scale = 5;
	Window win = Window("Depth of Field", scale * dim, scale * dim, glm::vec4(0, 1, 1, 0));

	Shader shader = Shader();
	shader.addVertexShader("res/RectVS.c");
	shader.addFragmentShader("res/RectFS.c");
	shader.compileShader();

	int width, height, num_channels;
	unsigned char* img = NULL;
	Texture::load_data("res/sat256.bmp", &img, &width, &height, &num_channels);

#if 0
	Texture::load_data("res/lena_256.png", &img, &width, &height, &num_channels);
	unsigned int N = width;
	//TODO: why malloc
	unsigned int* r = (unsigned int*)malloc(N * N * sizeof(int));
	unsigned int* g = (unsigned int*)malloc(N * N * sizeof(int));
	unsigned int* b = (unsigned int*)malloc(N * N * sizeof(int));

	deinterleave(img, N * N, &r, &g, &b);
	summed_area_table(r, N);
	summed_area_table(g, N);
	summed_area_table(b, N);

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			unsigned int tid = i * width + j;
			unsigned char* pixel = img + tid * num_channels;
			/*pixel[0] = (float)(r[tid]) / (float)(255 * 512 * 512) * 255.0;
			pixel[1] = (float)(g[tid]) / (float)(255 * 512 * 512) * 255.0;
			pixel[2] = (float)(b[tid]) / (float)(255 * 512 * 512) * 255.0;*/
			pixel[0] = (float)(r[tid]) / (float)(255 * dim * dim) * 255.0;
			pixel[1] = (float)(g[tid]) / (float)(255 * dim * dim) * 255.0;
			pixel[2] = (float)(b[tid]) / (float)(255 * dim * dim) * 255.0;
		}
	}
	
	free(r);
	free(g);
	free(b);

	std::ofstream myfile;
	myfile.open("res/or_sat256.csv");

	for (int r = height - 1; r >= 0; r--)
	{
		for (int c = 0; c < width; c++)
		{
			unsigned char* pix = img + (c + r * width) * num_channels;

			myfile << "(";
			for (int i = 0; i < num_channels; i++)
				myfile << (int)pix[i] << ":";
			myfile << "),";
		}
		myfile << "\n";
	}
	myfile.close();
#endif

	Texture texture = Texture(img, width, height, num_channels);
	//texture.write_to_bmp("res/sat256.bmp");

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
	shader.setUniform1f("width", texture.getWidth()); //don't worry about scaling, it's gonna end up as [0,1] anyways
	shader.setUniform1f("height", texture.getHeight());
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
		shader.setUniform2f("eye_pos", glm::vec2(x_transformed, y_transformed));

		//glActiveTexture(GL_TEXTURE0);
		texture.bind();
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
		//texture.unbind();

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