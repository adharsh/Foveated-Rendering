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

layout(std430, binding = 2) buffer SAT_Frame
{
	float frame[SIZE];
};

//	0 2 5 14 41
void main()
{
	fragColor = vec4(0);

	//after calculations in [-1,1] coordinates with eye_pos and pos
	//...+1.0f)/2.0f to get back to uv coordinates [0,1] to get texture color with texture 
	//https://learnopengl.com/Getting-started/Textures
	//https://stackoverflow.com/questions/45613310/switching-from-texture-to-texelfetch

	//cast these into int vectors, force into range [imagedim,imagedim]
	//coord is [-imagedim/2, imagedim/2], pos is [-1,1], uv is [0,1]
	//converting from pos to coord
	ivec2 pos_coord = ivec2((uv.x*(width - 1.0f) - width / 2.0f), (uv.y*(height - 1.0f) - height / 2.0f));
	ivec2 eye_coord = ivec2(((eye_pos.x + 1.0f) / 2.0f*(width - 1.0f) - width / 2.0f), ((eye_pos.y + 1.0f) / 2.0f*(height - 1.0f) - height / 2.0f));
	ivec2 dist_coord = pos_coord - eye_coord;

	//like before, copy and paste into new visual studio and try out if it works by value
	//also for debugging render each variable
	ivec2 id = ivec2(MAX_BOXES - 1, MAX_BOXES - 1);
	while (id.x >= 0 && abs(dist_coord.x) < box_coords[id.x])
		id.x--;
	while (id.y >= 0 && abs(dist_coord.y) < box_coords[id.y])
		id.y--;
	//dist_coord >= box_coords, shows the least value (used for 

	int max_id = max(id.x, id.y);
	int box_coord = box_coords[max_id];
	int box_dim = box_dims[max_id];

	ivec2 lowerleft_coord;
	if (id.x == id.y)
	{
		lowerleft_coord = eye_coord + box_coord;

		if (dist_coord.x < 0)
			lowerleft_coord.x = (lowerleft_coord.x - eye_coord.x + box_dim) * -1 + eye_coord.x;

		if (dist_coord.y < 0)
			lowerleft_coord.y = (lowerleft_coord.y - eye_coord.y + box_dim) * -1 + eye_coord.y;
	}
	else
	{
		//TODO, simplify below into one line if possible
		if (abs(dist_coord.x) > abs(dist_coord.y))
		{
			if (dist_coord.x < 0)
			{
				//left
				lowerleft_coord = ivec2(eye_coord.x - box_coord - box_dim, eye_coord.y - box_dim / 2);
			}
			else
			{
				//right
				lowerleft_coord = ivec2(eye_coord.x + box_coord, eye_coord.y - box_dim / 2);
			}
		}
		else
		{
			if (dist_coord.y < 0)
			{
				//down
				lowerleft_coord = ivec2(eye_coord.x - box_coord, eye_coord.y - box_dim / 2 - box_dim);
			}
			else
			{
				//up
				lowerleft_coord = ivec2(eye_coord.x - box_coord, eye_coord.y + box_dim / 2);
			}
		}

	}

	if (box_dim == 1)
	{
		lowerleft_coord = pos_coord;
	}

	//lowerleft_coord = pos_coord;
	//box_dim = 1;

	//todo: condense above code

	//zero indexed
	ivec2 lowerleft_idx = ivec2(lowerleft_coord.x + width / 2, lowerleft_coord.y + height / 2);
	ivec2 upperright_idx = ivec2(lowerleft_coord.x + box_dim + width / 2, lowerleft_coord.y + box_dim + height / 2);
	upperright_idx.x = upperright_idx.x >= width ? int(width) - 1 : upperright_idx.x;
	upperright_idx.y = upperright_idx.y >= height ? int(height) - 1 : upperright_idx.y;

	//A < D
	int A = lowerleft_idx.x + int(width) * lowerleft_idx.y;
	int B = upperright_idx.x + int(width) * lowerleft_idx.y;
	int C = lowerleft_idx.x + int(width) * upperright_idx.y;
	int D = upperright_idx.x + int(width) * upperright_idx.y;

	//pixel[1] = (float)(g[tid]) / (float)(255 * dim * dim) * 255.0;
	//255 * 256 * 256
	fragColor = vec4(frame[D] - frame[C] - frame[B] + frame[A],
			 		 frame[D + int(width) * int(height)] - frame[C + int(width) * int(height)] - frame[B + int(width) * int(height)] + frame[A + int(width) * int(height)],
			 		 frame[D + 2 * int(width) * int(height)] - frame[C + 2 * int(width) * int(height)] - frame[B + 2 * int(width) * int(height)] + frame[A + 2 * int(width) * int(height)],
					 0);
	fragColor /= 255.0f;
	fragColor.w = 1;

	//if (lower_pos == uv)
	//	fragColor = vec4(1.0f);
	//fragColor = vec4((eye_pos.xy + 1.0f) / 2.0f, 0.0f, 1.0f);
	//fragColor = vec4((pos.xy + 1.0f) / 2.0f, 0.0f, 1.0f);

	//fragColor = vec4((dist_coord.x + width / 2) / (width - 1), (dist_coord.y + height / 2) / (height - 1), 0.0f, 1.0f);
	//fragColor = vec4((eye_coord.x + width / 2) / (width - 1), (eye_coord.y + height / 2) / (height - 1), 0.0f, 1.0f);
	//fragColor =  vec4((pos_coord.x + width / 2.0f) / (width - 1.0f), (pos_coord.y + height / 2.0f) / (height - 1), 0.0f, 1.0f);

	//fragColor = vec4(id.xy/9.0f, 0.0f, 1.0f);
	//fragColor = vec4(max_id / 9.0f);
	//fragColor = vec4((box_coord + width / 2.0f) / (width - 1.0f), (box_coord + height / 2.0f) / (height - 1), 0.0f, 1.0f);
	//fragColor = vec4(box_dim/41.0f);
	
	//fragColor = vec4((lowerleft_coord.x + width / 2.0f) / (width - 1.0f), (lowerleft_coord.y + height / 2.0f) / (height - 1), 0.0f, 1.0f);
	//fragColor = vec4((upperright_coord.x + width / 2.0f) / (width - 1.0f), (upperright_coord.y + height / 2.0f) / (height - 1), 0.0f, 1.0f);

	
	//Good one for demo purposes
	/*if (id.x != id.y)
		fragColor = vec4(1);
	else
		fragColor = vec4(0);
	*/
}