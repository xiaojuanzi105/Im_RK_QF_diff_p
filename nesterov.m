function NAG=nesterov(h, N, z0, d, W, H, L, SW)
y=zeros(d,N+1);
x=zeros(d,N+1);
x(:,1) = z0(1:d);
y(:,1)=x(:,1);
for i=2:N+1
if SW==1
  x(:,i)=x(:, i-1) -h*grad_Fu(x(:, i-1) , W, H, L);
else
    y(:,i)=x(:, i-1) -h*grad_Fu(x(:, i-1) , W, H, L);
x(:,i)=y(:,i)+0.8*(y(:,i)-y(:,i-1));
end
end
NAG=x;
end