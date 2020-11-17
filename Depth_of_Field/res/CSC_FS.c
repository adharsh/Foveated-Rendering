#version 430 core
in vec2 uv;
in vec2 pos;
out vec4 fragColor;

uniform vec2 eye_pos; //[-1,1]
uniform float width;
uniform float height;

//#define resolution 4

layout(std430, binding = 2) buffer SAT_Frame
{
	float frame[];
};

vec3 sum_SAT(ivec2 lowerleft_idx, ivec2 dim)
{
	ivec2 upperright_idx = ivec2(lowerleft_idx.x + dim.x, lowerleft_idx.y + dim.y);

	//A < D
	int A = lowerleft_idx.x + int(width) * lowerleft_idx.y;
	int B = upperright_idx.x + int(width) * lowerleft_idx.y;
	int C = lowerleft_idx.x + int(width) * upperright_idx.y;
	int D = upperright_idx.x + int(width) * upperright_idx.y;

	vec3 sum = vec3(frame[D] - frame[C] - frame[B] + frame[A],
		frame[D + int(width) * int(height)] - frame[C + int(width) * int(height)] - frame[B + int(width) * int(height)] + frame[A + int(width) * int(height)],
		frame[D + 2 * int(width) * int(height)] - frame[C + 2 * int(width) * int(height)] - frame[B + 2 * int(width) * int(height)] + frame[A + 2 * int(width) * int(height)]
	);

	return sum;
}

//	0 2 5 14 41
void main()
{
	ivec2 pos_idx = ivec2((uv.x*(width - 1.0f)), (uv.y*(height - 1.0f)));
	//vec3 sum = sum_SAT(pos_idx, ivec2(1, 1));
	
	int N = 30;

	// Range: [-width/2, width/2], [-height/2, height/2]
	
	//ivec2 eye_coord = ivec2(((eye_pos.x + 1.0f) / 2.0f*(width - 1.0f) - width / 2.0f), ((eye_pos.y + 1.0f) / 2.0f*(height - 1.0f) - height / 2.0f));
	//ivec2 dist_coord = pos_coord - eye_coord;

	float sqrt_2 = 1.41421356237f;

	float num_pixels = 0;
	float r = float(int(N / 2));
	// Middle square
	float square_dim = r / sqrt_2 * 2;
	num_pixels += square_dim * square_dim;
	vec3 sum = vec3(0);
	sum += sum_SAT(pos_idx + ivec2(-square_dim/2, -square_dim/2), ivec2(square_dim, square_dim));
	//num_pixels += 1;
	//sum += sum_SAT(pos_coord, vec2(1, 1));


	//edges
#if 1
	float step = 1;
	int num_siderects = int(r - square_dim / 2.0f);
	for (int i = 1; i <= num_siderects; i++)
	{
		for (int j = 1; j <= 4; j++)
		{
			switch (j)
			{
				//top
				case 1: sum += sum_SAT(pos_idx + ivec2(-1 * r + (i - 1) * step, r + i - 1), ivec2(square_dim - step * 2 * i, 1)); break;
				//bottom
				case 2: sum += sum_SAT(pos_idx + ivec2(-1 * r + (i - 1) * step, -1 * r - (i - 1) ), ivec2(square_dim - step * 2 * i, 1)); break;
				//right
				case 3: sum += sum_SAT(pos_idx + ivec2(r + i - 1 , -1 * r + (i - 1) * step), ivec2(1, square_dim - step * 2 * i)); break;
				//left
				case 4: sum += sum_SAT(pos_idx + ivec2(-1 * r - (i - 1), -1 * r + (i - 1) * step), ivec2(1, square_dim - step * 2 * i)); break;
			}
		}
		num_pixels += 4 * square_dim - step * 2 * i;
	}
#endif

	fragColor = vec4(sum.xyz / 255 / num_pixels, 1);
}