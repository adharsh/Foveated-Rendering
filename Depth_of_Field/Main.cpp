#include "Shader.h"
#include "Window.h"
#include "Texture.h"
#include "cuda/deinterleave.cuh"
#include "cuda/summed_area_table.cuh"

#include <fstream>
#include <iostream>

int main()
{
	Window win = Window("Depth of Field", 3 * 512, 3 * 512, glm::vec4(0, 1, 1, 0));
	
	Shader shader = Shader();
	shader.addVertexShader("res/RectVS.txt");
	shader.addFragmentShader("res/RectFS.txt");
	shader.compileShader();

	int width, height, num_channels;
	unsigned char* img = NULL;
	Texture::load_data("res/lena_256.png", &img, &width, &height, &num_channels);

	unsigned int N = width;


	unsigned int* r = (unsigned int*)malloc(N * N * sizeof(int));
	unsigned int* g = (unsigned int*)malloc(N * N * sizeof(int));
	unsigned int* b = (unsigned int*)malloc(N * N *sizeof(int));
	
	deinterleave(img, N * N, &r, &g, &b);
	summed_area_table(r, N);
	summed_area_table(g, N);
	summed_area_table(b, N);

	/*std::ofstream myfile;
	myfile.open("red.csv");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			myfile << r[i * N + j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();

	myfile.open("blue.csv");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			myfile << b[i * N + j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();

	myfile.open("green.csv");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			myfile << g[i * N + j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();*/

	std::cout << r[N * N - 1] << " " << g[N * N - 1] << " " << b[N * N - 1] << " " << std::endl;

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			unsigned int tid = i * width + j;
			unsigned char* pixel = img + tid * num_channels;
			pixel[0] = (float)(r[tid]) / (float)(255 * 512 * 512) * 255.0;
			pixel[1] = (float)(g[tid]) / (float)(255 * 512 * 512) * 255.0;
			pixel[2] = (float)(b[tid]) / (float)(255 * 512 * 512) * 255.0;
		}
	}

	free(r);
	free(g);
	free(b);

	Texture texture = Texture(img, width, height, num_channels);
	
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
	while (!win.closed())
	{
		win.clear();

		shader.bind();
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

		//glActiveTexture(GL_TEXTURE0);
		texture.bind();
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
		texture.unbind();
		
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