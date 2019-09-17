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

	//*1* \/ \/ \/ below id is always same why
	//like before, copy and paste into new visual studio and try out if it works by value
	//also for debugging render each variable
	//somehow always 9, doesn't change
	//pos_coord is right, but box_coords is not working
	ivec2 id = ivec2(N-1, N-1);
	while(id.x >= 0 && abs(dist_coord.x) < box_coords[id.x])
		id.x--;
	while(id.y >= 0 && abs(dist_coord.y) < box_coords[id.y])
		id.y--;

	ivec2 box_coord = ivec2( box_coords[id.x], box_coords[id.y] ); //why are box_coord and box_dim different
	
	//	0 2 5 14 41, lowerleft id is at value
	ivec2 box_dim;
	if(id.x > 1)
		box_dim.x = 3 * (box_coord.x - box_coords[id.x - 1]);
	else if(id.x == 1)
		box_dim.x = box_coords[1] + 1; //this can be adjusted, adjusting box_coords[1] will make center bigger as well
	else
		box_dim.x = box_coords[0]; //should always be zero, box_dim should be 0 for center
	if(id.y > 1)
		box_dim.y = 3 * (box_coord.y - box_coords[id.y - 1]);
	else if(id.y == 1)
		box_dim.y = box_coords[1] + 1; //this can be adjusted, adjusting box_coords[1] will make center bigger as well
	else
		box_dim.y = box_coords[0]; //should always be zero, box_dim should be 0 for center
	
	//%%%%%%%%%%%%
	//*2*
	//check all of this
	ivec2 lowerleft_coord;
	if(abs(dist_coord.x) > 1)
	{
		lowerleft_coord.x = box_coord.x + eye_coord.x;
		if(dist_coord.x < 0)
			lowerleft_coord.x = -(box_dim.x - lowerleft_coord.x); //probably this check
	}
	else
	{
	//for other values, hardcode conditions (without hardcoding too much that scaling isn't possible)
		if(abs(dist_coord.y) > 1) //vertical line outside of center
		{
			lowerleft_coord.x = eye_coord.x - box_dim.x / 2; //rounding?
		}
		else //very center
		{
			lowerleft_coord.x = pos_coord.x;
		}
	}

	if(abs(dist_coord.y) > 1)
	{
		lowerleft_coord.y = box_coord.y + eye_coord.y;
		if(dist_coord.y < 0)
			lowerleft_coord.y = -(box_dim.y - lowerleft_coord.y); //probably this check
	}
	else
	{
		if(abs(dist_coord.x) > 1) //horizontal line outside of center
		{
			lowerleft_coord.y = eye_coord.y - box_dim.y/2; //rounding + 0.5f?
		}
		else //very center
		{
			lowerleft_coord.y = pos_coord.y;
		}

	}
	//condense above code^
	//make sure lowerleft_coord is already int by her
	// find a way to find lowerleft_coord
	// x will go completely to the left always, equivalent to the box_coord x val
	// y will go down as much as it needs to always
	////should be readjusted to lower left as well as pos [-1,1]
	vec2 lowerleft_pos = vec2( (float(lowerleft_coord.x) + width/2.0f)/(width-1.0f), (float(lowerleft_coord.y) + height/2.0f)/(height-1.0f));
	vec2 upperright_pos = vec2( (float(lowerleft_coord.x) + box_dim.x + width/2.0f)/(width-1.0f), (float(lowerleft_coord.y) + box_dim.y + height/2.0f)/(height-1.0f));
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

	
	//if (lower_pos == uv)
	//	fragColor = vec4(1.0f);
	//fragColor = texture(texture_sampler, upperright_pos - lowerleft_pos);
	//fragColor = vec4((eye_pos.xy + 1.0f) / 2.0f, 0.0f, 1.0f);
	//fragColor = vec4((pos.xy + 1.0f) / 2.0f, 0.0f, 1.0f);

	//fragColor = vec4((dist_coord.x + width / 2) / (width - 1), (dist_coord.y + height / 2) / (height - 1), 0.0f, 1.0f);
	//fragColor = vec4((eye_coord.x + width / 2) / (width - 1), (eye_coord.y + height / 2) / (height - 1), 0.0f, 1.0f);
	//fragColor =  vec4((pos_coord.x + width / 2.0f) / (width - 1.0f), (pos_coord.y + height / 2.0f) / (height - 1), 0.0f, 1.0f);

	fragColor = vec4(id.xy/9.0f, 0.0f, 1.0f);
	//fragColor = vec4(box_dim.xy/3.0f, 0.0f, 1.0f);
	
	//fragColor = vec4(lowerleft_pos.xy, 0.0f, 1.0f);
	
	//fragColor = vec4((float(pos_coord.x) + width / 2.0f) / (width - 1.0f), (float(pos_coord.y) + height / 2.0f) / (height - 1.0f), 0.0f, 1.0f);

	//if (pos_coord != vec2(0, 0))
	//	fragColor = vec4(0.0f, 0.0f, 1.0f, 0.0f);
	//else
	//	fragColor = vec4(1.0f, 0.0f, 0.0f, 0.0f);
	//pos_coord = vec2((pos_coord.x + width / 2) / (width - 1), (pos_coord.y + height / 2) / (height - 1));
	 
	//fragColor = vec4((eye_coord.x + width / 2) / (width - 1), (eye_coord.y + height / 2) / (height - 1), 0.0f, 1.0f);
}