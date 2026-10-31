// max_pool2d_grad scalar implementation.
int input_width=P(5);int input_height=P(4);int plane=i/(input_width*input_height);int iy=(i/input_width)%input_height;int ix=i%input_width;int output_height=P(30);int output_width=P(31);float value=0.0f;
for(int oy=0;oy<output_height;++oy)for(int ox=0;ox<output_width;++ox){int start_y=oy*P(66)-P(68);int start_x=ox*P(67)-P(69);
    if(iy>=start_y&&iy<start_y+P(64)&&ix>=start_x&&ix<start_x+P(65)){
        float maximum_value=-EL_INF;int selected=-1;
        for(int ky=0;ky<P(64);++ky)for(int kx=0;kx<P(65);++kx){int y=start_y+ky;int x=start_x+kx;if(y>=0&&y<input_height&&x>=0&&x<input_width){int index=(plane*input_height+y)*input_width+x;float candidate=A(index);if(selected<0||candidate>maximum_value){maximum_value=candidate;selected=index;}}}
        if(selected==i)value+=B((plane*output_height+oy)*output_width+ox);
    }
}
W(i,value);
