// leaky_relu scalar implementation.
float x=A(i);
W(i,x>=0.0f?x:F(64)*x);
