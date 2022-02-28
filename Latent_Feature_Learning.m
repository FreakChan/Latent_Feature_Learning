function [W] = Latent_Feature_Learning(X,Y,L,lambda1,lambda2,lambda3,lambda4,p,h,Ite)

if (~exist('lambda1','var'))    lambda1 = 1;    end
if (~exist('lambda2','var'))    lambda2 = 10;   end
if (~exist('lambda3','var'))    lambda3 = 1;    end
if (~exist('lambda4','var'))    lambda4 = 10;   end
if (~exist('p','var'))          p = 1;          end
if (~exist('h','var'))          h = 500;          end
if (~exist('Ite','var'))        Ite = 50;       end
if (~exist('L','var'))
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 3;
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    L = constructW(X,options);
end



% initial
[ins fea] = size(X);
class = size(Y,2);

V = ones(fea,h);
H = X*V;   % ins * h
I1 = eye(ins);

d = ones(h,1);
dv = ones(fea,1);

for iter = 1:Ite     
    D = diag(d);
    DV = diag(dv);
    
    % fix H & V, update W
    W = (H'*H + lambda1*H'*L*H + lambda2*D)\ (H'*Y);    % h * 3
    W21 = sqrt(sum(W.*W,2))+ eps; 
    d = 0.5./W21;
   
    % fix H & W, update V
    temp1 = inv(X'*X + (lambda4/lambda3)*DV);
    V = temp1 * (X'*H);    % fea * h
    
    V21 = sqrt(sum(V.*V,2))+ eps;    
    dv = 0.5./V21;  
    
    % fix V & W, update H
    temp = inv(2 * lambda1 * L + I1);
    A = 2 * lambda3 * temp;
    B = W * W';
    C = -temp * (Y * W' + 2 * lambda3 * X * V);
    H = lyap(A,B,C);   % ins * h
    
      
     obj(iter) = norm(Y-H*W,'fro')^2 + lambda1* trace(W'*H'*L*H*W) + lambda2*sum(W21) + lambda3 *norm(X*V-H,'fro')^2 + lambda4*sum(V21);
end
 plot(obj)
end


