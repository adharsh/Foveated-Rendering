#include "Shader.h"
#include "Window.h"

int main()
{
	int s = 3;
	Window win = Window("Depth of Field", s * 512, s * 512, glm::vec4(0, 1, 1, 0));
	Shader tri = Shader();
	tri.addVertexShader("shaders/RectVS.txt");
	tri.addFragmentShader("shaders/RectFS.txt");
	tri.compileShader();

	float vertices[] = {
		 1, 1, 0,   // top right
		 1, -1, 0,  // bottom right
		-1, -1, 0,  // bottom left
		-1,  1, 0   // top left 
	};
	unsigned int indices[] = {  // note that we start from 0!
		0, 1, 3,  // first Triangle
		1, 2, 3   // second Triangle
	};
	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	while (!win.closed())
	{
		win.clear();

		tri.bind();
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		tri.unbind();

		win.update();
	}

}