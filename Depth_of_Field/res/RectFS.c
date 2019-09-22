#version 430 core
#define SIZE (256*256*3)
#define MAX_BOXES 10
in vec2 uv;
in vec2 pos;
out vec4 fragColor;

uniform vec2 eye_pos; //[-1,1]
uniform int box_coords[MAX_BOXES];
uniform int box_dims[MAX_BOXES];
uniform float width;
uniform float height;
//uniform int resolution;
#define resolution 4

#define RECURS 0

layout(std430, binding = 2) buffer SAT_Frame
{
	float frame[SIZE];
};

//	0 2 5 14 41
void main()
{
	fragColor = vec4(0);
	int recurs = resolution;

	ivec2 pos_coord = ivec2((uv.x*(width - 1.0f) - width / 2.0f), (uv.y*(height - 1.0f) - height / 2.0f));
	ivec2 eye_coord = ivec2(((eye_pos.x + 1.0f) / 2.0f*(width - 1.0f) - width / 2.0f), ((eye_pos.y + 1.0f) / 2.0f*(height - 1.0f) - height / 2.0f));
	ivec2 dist_coord = pos_coord - eye_coord;
	ivec2 orig_eye_coord = eye_coord;

	ivec2 id = ivec2(0, 0);
	while (id.x < MAX_BOXES && box_coords[id.x] <= abs(dist_coord.x))
		id.x++;
	while (id.y < MAX_BOXES && box_coords[id.y] <= abs(dist_coord.y))
		id.y++;
	id -= 1;

	int max_id = max(id.x, id.y);
	int box_coord = box_coords[max_id];
	int box_dim = box_dims[max_id];

	ivec2 lowerleft_coord;

	//#if RECURS
	while (recurs > 0 && box_dim > 0 && max_id >= 0)
	{
		if(abs(dist_coord.x) < box_coord && abs(dist_coord.y) < box_coord) //center
		{
			lowerleft_coord = eye_coord - box_dim/2;
			//recurs = 0;
			fragColor = vec4(1, 0, 0, 1);
		}
		else 
//#endif
		if(abs(dist_coord.x) > box_coord && abs(dist_coord.y) > box_coord) //corners
		{
			fragColor = vec4(0, 1, 0, 1);
			//fragColor = vec4(1);
			lowerleft_coord = eye_coord + box_coord; //top right

			if (dist_coord.x > box_coord && dist_coord.y > box_coord) //top right
			{
				fragColor = vec4(0, 0.2, 0, 1);
			}
			else if (dist_coord.x < -box_coord  && dist_coord.y < -box_coord) //bottom left
			{
				lowerleft_coord.x = (box_coord + box_dim) * -1 + eye_coord.x; 
				lowerleft_coord.y = (box_coord + box_dim) * -1 + eye_coord.y;
				fragColor = vec4(0, 0.6, 0, 1);
			}
			else if (dist_coord.x < -box_coord  && dist_coord.y > box_coord) //top left
			{
				lowerleft_coord.x = (box_coord + box_dim) * -1 + eye_coord.x;
				fragColor = vec4(0, 0.4, 0, 1);
			}
			else //bottom right
			{
				lowerleft_coord.y = (box_coord + box_dim) * -1 + eye_coord.y;
				fragColor = vec4(0, 0.8, 0, 1);
			}
				
		}
		else
		{
			//TODO, simplify below into one line if possible
			//split into more explicit cases without using abs
			if (abs(dist_coord.x) > abs(dist_coord.y))
			{
				if (dist_coord.x < -box_coord) //left
				{
					lowerleft_coord = ivec2(eye_coord.x - box_coord - box_dim, eye_coord.y - box_dim / 2);
					fragColor = vec4(0, 0, 0.2, 1);
				}
				else//right
				{
					lowerleft_coord = ivec2(eye_coord.x + box_coord, eye_coord.y - box_dim / 2);
					fragColor = vec4(0, 0, 0.4, 1);
				}
					
			}
			else
			{
				if (dist_coord.y < -box_coord) //bottom
				{
					lowerleft_coord = ivec2(eye_coord.x - box_dim / 2, eye_coord.y - box_coord - box_dim);
					fragColor = vec4(0, 0, 0.6, 1);
				}
				else //top
				{
					lowerleft_coord = ivec2(eye_coord.x - box_dim / 2, eye_coord.y + box_coord);
					fragColor = vec4(0, 0, 1, 1);
				}
					
			}
		}
		//#if RECURS		
		eye_coord = lowerleft_coord + box_dim / 2; //eye_coord not really eye_coord anymore, now center of subdivision
		dist_coord = pos_coord - eye_coord;

		//*1*
		box_coord = box_coords[--max_id];
		box_dim /= 3; //1/3 -> 0 then returns //box_dim = box_dims[max_id];
		recurs--;
	}

	box_dim = box_dim == 0 ? 1 : box_dim * 3;
	//box_dim

	ivec2 center_check_coord = abs(pos_coord - orig_eye_coord);
	int center_bounds = box_coords[resolution];
	if (center_check_coord.x < center_bounds && center_check_coord.y < center_bounds)
	{
		lowerleft_coord = pos_coord;
		box_dim = 1;
	}
	
	//todo: to toggle foveated_rendering on/off
	//lowerleft_coord = pos_coord;
	//box_dim = 1;

#if 1
	ivec2 lowerleft_idx = ivec2(lowerleft_coord.x + width / 2, lowerleft_coord.y + height / 2);
	lowerleft_idx.x = lowerleft_idx.x < 0 ? 0 : lowerleft_idx.x;
	lowerleft_idx.y = lowerleft_idx.y < 0 ? 0 : lowerleft_idx.y;
	//lowerleft_idx.x = lowerleft_idx.x >= width ? int(width) - 1 : lowerleft_idx.x;
	//lowerleft_idx.y = lowerleft_idx.y >= height ? int(height) - 1 : lowerleft_idx.y;

	ivec2 upperright_idx = ivec2(lowerleft_coord.x + box_dim + width / 2, lowerleft_coord.y + box_dim + height / 2);
	//upperright_idx.x = upperright_idx.x < 0 ? 0 : upperright_idx.x;
	//upperright_idx.y = upperright_idx.y < 0 ? 0 : upperright_idx.y;
	upperright_idx.x = upperright_idx.x >= width ? int(width) - 1 : upperright_idx.x;
	upperright_idx.y = upperright_idx.y >= height ? int(height) - 1 : upperright_idx.y;
#else
	ivec2 lowerleft_idx = ivec2(lowerleft_coord.x + width / 2, lowerleft_coord.y + height / 2);
	ivec2 upperright_idx = ivec2(lowerleft_coord.x + box_dim + width / 2, lowerleft_coord.y + box_dim + height / 2);
	lowerleft_idx.x = lowerleft_idx.x < 0 ? 0 : lowerleft_idx.x;
	lowerleft_idx.y = lowerleft_idx.y < 0 ? 0 : lowerleft_idx.y;
	upperright_idx.x = upperright_idx.x >= width ? int(width) - 1 : upperright_idx.x;
	upperright_idx.y = upperright_idx.y >= height ? int(height) - 1 : upperright_idx.y;
#endif

	//A < D
	int A = lowerleft_idx.x + int(width) * lowerleft_idx.y;
	int B = upperright_idx.x + int(width) * lowerleft_idx.y;
	int C = lowerleft_idx.x + int(width) * upperright_idx.y;
	int D = upperright_idx.x + int(width) * upperright_idx.y;

#if 1
	vec3 fragColor_tmp = vec3(frame[D] - frame[C] - frame[B] + frame[A],
		frame[D + int(width) * int(height)] - frame[C + int(width) * int(height)] - frame[B + int(width) * int(height)] + frame[A + int(width) * int(height)],
		frame[D + 2 * int(width) * int(height)] - frame[C + 2 * int(width) * int(height)] - frame[B + 2 * int(width) * int(height)] + frame[A + 2 * int(width) * int(height)]
		);
	fragColor = vec4(normalize(fragColor_tmp.xyz), 1);

	//fragColor = vec4((fragColor_tmp.xyz) / 255, 1);
	
	
	
	//int max_color = int(max(max(fragColor_tmp.x, fragColor_tmp.y), fragColor_tmp.z));
	//fragColor = vec4(fragColor_tmp.xyz / max_color, 1);
	//fragColor = normalize(vec4(fragColor_tmp.xyz, 1))
	
	//https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
	//fragColor_tmp *= vec3(0.2126, 0.7152, 0.0722);
	//fragColor = vec4(normalize(fragColor_tmp.xyz), 1);
	
	//fragColor.y *= 1;
	//fragColor = vec4(normalize(fragColor_tmp).xyz, 1);
	//fragColor = vec4(fragColor_tmp.xyz/255, 1);
#else
	fragColor = vec4((lowerleft_coord.x + width / 2.0f) / (width - 1.0f), (lowerleft_coord.y + height / 2.0f) / (height - 1), 0.0f, 1.0f);
#endif
	
	
}