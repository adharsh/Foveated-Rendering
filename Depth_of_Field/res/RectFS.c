#version 330 core
#define N 10
in vec2 uv;
in vec2 pos;
out vec4 fragColor;

uniform sampler2D texture_sampler;
uniform vec2 eye_pos; //[-1,1]
uniform int box_coords[N];
uniform float width;
uniform float height;

void main()
{
	fragColor = vec4(0);
//fragColor = texture(texture_sampler, uv);
//
////float val = 0.5f;
//if( eye_pos.x - val < pos.x && pos.x < eye_pos.x + val &&
//	eye_pos.y - val < pos.y && pos.y < eye_pos.y + val)
//	fragColor = texture(texture_sampler, (eye_pos+1.0f)/2.0f);
//	//fragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	
	//after calculations in [-1,1] coordinates with eye_pos and pos
	//...+1.0f)/2.0f to get back to uv coordinates [0,1] to get texture color with texture 
	//https://learnopengl.com/Getting-started/Textures
	//https://stackoverflow.com/questions/45613310/switching-from-texture-to-texelfetch
	
	//cast these into int vectors, force into range [imagedim,imagedim]
	//coord is [-imagedim/2, imagedim/2], pos is [-1,1], uv is [0,1]
	//converting from pos to coord
	ivec2 pos_coord = ivec2((uv.x*(width - 1.0f) - width / 2.0f), (uv.y*(height - 1.0f) - height / 2.0f));
	ivec2 eye_coord = ivec2( ((eye_pos.x + 1.0f)/2.0f*(width - 1.0f) - width/2.0f), ((eye_pos.y + 1.0f)/2.0f*(height - 1.0f) - height/2.0f) );
	ivec2 dist_coord = pos_coord - eye_coord;

	//like before, copy and paste into new visual studio and try out if it works by value
	//also for debugging render each variable
	ivec2 id = ivec2(N-1, N-1);
	while(id.x >= 0 && abs(dist_coord.x) < box_coords[id.x])
		id.x--;
	while(id.y >= 0 && abs(dist_coord.y) < box_coords[id.y])
		id.y--;
	//dist_coord >= box_coords, shows the least value (used for 

	int max_id = max(id.x, id.y);
	int box_coord = box_coords[max_id];
	
	//	0 2 5 14 41, lowerleft id is at value
	//id.x == 1; this can be adjusted, adjusting box_coords[1] will make center bigger as well
		//id.x == 0; should always be zero, box_dim should be 0 for center
	int box_dim;
	if (max_id > 1)
		box_dim = 3 * (box_coord - box_coords[max_id - 1]);
	else
		box_dim = box_coords[max_id] + 1; 

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
				lowerleft_coord = ivec2(eye_coord.x - box_coord - box_dim, eye_coord.y - box_dim/2);
			}
			else
			{
				//right
				lowerleft_coord = ivec2(eye_coord.x + box_coord, eye_coord.y - box_dim/2);
			}
		}
		else
		{
			if (dist_coord.y < 0)
			{
				//down
				lowerleft_coord = ivec2(eye_coord.x - box_coord, eye_coord.y - box_dim/2 - box_dim);
			}
			else
			{
				//up
				lowerleft_coord = ivec2(eye_coord.x - box_coord, eye_coord.y + box_dim/2);
			}
		}

	}
	lowerleft_coord = pos_coord;
	box_dim = 1;

	//condense above code^
	//make sure lowerleft_coord is already int by her
	// find a way to find lowerleft_coord
	// x will go completely to the left always, equivalent to the box_coord x val
	// y will go down as much as it needs to always
	////should be readjusted to lower left as well as pos [-1,1]
	vec2 lowerleft_pos = vec2( (float(lowerleft_coord.x) + width/2.0f)/(width-1.0f), (float(lowerleft_coord.y) + height/2.0f)/(height-1.0f));
	vec2 upperright_pos = vec2( (float(lowerleft_coord.x) + box_dim + width/2.0f)/(width-1.0f), (float(lowerleft_coord.y) + box_dim + height/2.0f)/(height-1.0f));
	upperright_pos.x = upperright_pos.x > 1.0f? 1.0f : upperright_pos.x;
	upperright_pos.y = upperright_pos.y > 1.0f? 1.0f : upperright_pos.y;

	vec2 A = lowerleft_pos;
	vec2 B = vec2(upperright_pos.x, lowerleft_pos.y);
	vec2 C = vec2(lowerleft_pos.x, upperright_pos.y);
	vec2 D = upperright_pos;

	//pixel[1] = (float)(g[tid]) / (float)(255 * dim * dim) * 255.0;
	//255 * 256 * 256
	fragColor = texture(texture_sampler, D) - texture(texture_sampler, B) - texture(texture_sampler, C) + texture(texture_sampler, A);
	fragColor.w = 1;

	fragColor *= 2;

	//fragColor = texture(texture_sampler, uv);
	//if (lower_pos == uv)
	//	fragColor = vec4(1.0f);
	//fragColor = texture(texture_sampler, upperright_pos - lowerleft_pos);
	//fragColor = vec4((eye_pos.xy + 1.0f) / 2.0f, 0.0f, 1.0f);
	//fragColor = vec4((pos.xy + 1.0f) / 2.0f, 0.0f, 1.0f);

	//fragColor = vec4((dist_coord.x + width / 2) / (width - 1), (dist_coord.y + height / 2) / (height - 1), 0.0f, 1.0f);
	//fragColor = vec4((eye_coord.x + width / 2) / (width - 1), (eye_coord.y + height / 2) / (height - 1), 0.0f, 1.0f);
	//fragColor =  vec4((pos_coord.x + width / 2.0f) / (width - 1.0f), (pos_coord.y + height / 2.0f) / (height - 1), 0.0f, 1.0f);

	//fragColor = vec4(id.xy/9.0f, 0.0f, 1.0f);
	//fragColor = vec4(max_id / 9.0f);
	//fragColor = vec4((box_coord + width / 2.0f) / (width - 1.0f), (box_coord + height / 2.0f) / (height - 1), 0.0f, 1.0f);
	//fragColor = vec4(box_dim/41.0f);
	
	//fragColor = vec4(lowerleft_pos.xy, 0.0f, 1.0f);
	//fragColor = vec4(upperright_pos.xy, 0.0f, 1.0f);

	
	//Good one for demo purposes
	/*if (id.x != id.y)
		fragColor = vec4(1);
	else
		fragColor = vec4(0);*/
}