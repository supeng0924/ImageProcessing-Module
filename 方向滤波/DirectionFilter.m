function [  ] = DirectionFilter( input_args )
%UNTITLED4 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%%
%����ͼ��
image=imread('Lena.jpg');
%��ʼ��ȫΪ0����imageͬ�ߴ��ͼ��
deal=zeros(size(image));
[rows,cols]=size(image);
%%
%�����������
sigma=10;
theta=0;
%����0:30:150 �ܹ�6��������˲�����ͼ�Ա�
%��ͼ�� x yΪ������ theta����x��������н�  theta��Χ��0-179
theta_range=0:30:150;
for theta_i=1:numel(theta_range)
    theta=theta_range(theta_i);
%%
%����Ȩֵ����
%Ȩֵ����
weight_length=4*sigma+1;
%Ȩֵ����һ��
half_length=floor(weight_length/2);
%Ȩֵ�����ʼ��
weight=zeros(1,weight_length);
%б��
k=tan(theta/180*pi);
%����t����
t_val=(1:weight_length)-half_length-1;
%����ʵ��dert_x dert_y �仯��
if theta==0
    dert_y=zeros(size(t_val));
    dert_x=-half_length:half_length;
elseif theta==90
    dert_y=-half_length:half_length;
    dert_x=zeros(size(t_val));
elseif (theta>0&&theta<45)||(theta>90&&theta<135)
    dert_y=t_val
    dert_x=t_val/k;
    dert_x=round(dert_x)
%     plot(act_x,act_y);axis equal
else
    dert_x=t_val
    dert_y=t_val*k;
    dert_y=round(dert_y)
%     plot(act_x,act_y);axis equal
end

distance=dert_y.*dert_y+dert_x.*dert_x;
weight=exp(-distance./(2*sigma*sigma));
%%
%���ص㴦��
for row=1:rows
    for col=1:cols
        %����ÿһ�����ص��ֵ
        
        %�ۼ����ص�ֵ
        pixel_total=0;
        %�ۼƸ���
        p_total=0;
        
        for i=1:weight_length
            %�����ʵ��ͼ���� x y ����
            act_x=dert_x(i)+col;
            act_y=dert_y(i)+row;
            %���ڳ���ͼ��Χ�ڵ����ص� ��ȥ
            if act_x>0&&act_x<=cols&&act_y>0&&act_y<=rows
                pixel_total=double(pixel_total)+double(image(act_y,act_x))*weight(i);
                p_total=p_total+weight(i);
            end
        end
        %�������ص������ֵ
        value=pixel_total/p_total;
        deal(row,col)=uint8(value);    
    end
end
%%
%���ͼ��
subplot(2,3,theta_i);imshow(deal,[]);title(num2str(theta));
end
end

