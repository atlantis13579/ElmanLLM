// normal scalar implementation.
float u=elman_random(2*i,P(64),P(65));
float v=elman_random(2*i+1,P(64),P(65));
W(i,F(66)+F(67)*EL_SQRT(-2.0f*EL_LOG(u))*EL_COS(6.283185307179586f*v));
