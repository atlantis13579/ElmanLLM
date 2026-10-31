// gelu scalar implementation.
float x=A(i);
W(i,0.5f*x*(1.0f+EL_TANH(0.7978845608028654f*(x+0.044715f*x*x*x))));
