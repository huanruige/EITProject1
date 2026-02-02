% 启动Eidors 按照自己电脑中路径配置 首次启动后可注释掉 提高速度
% run('D:\soft\eidors-v3.10-ng\eidors\startup.m')
run('D:\Program Files\eidors-v3.12-ng\eidors\startup.m');
% 模型创建 16 电极圆形均匀分布
n_elecs_per_plane_1 = 16;
th = linspace(0,360,n_elecs_per_plane_1+1)'; 
th(1) = [];
els = (th+67.5)*[1,0];
elec_sz = 0.025;
fmdl = ng_mk_cyl_models([0,20,0.6],els,[elec_sz,0,0.5]);
for i=1:n_elecs_per_plane_1
   fmdl.electrode(i).z_contact = 0.0001;
end
        
% 设置电压和电流刺激模式
fmdl.stimulation=mk_stim_patterns(16,1,[0,1],[0,1],{'no_meas_current','rotate_meas'},1);
imdl= mk_common_model('c2C',16);
imdl.fwd_model=fmdl;

% 构建和处理有限元网格模型
grid = linspace(-20, 20, 65);
[imdl.rec_model, imdl.fwd_model.coarse2fine] = ...
     mk_grid_model(imdl.fwd_model, grid, grid);
outside = find(sum(imdl.fwd_model.coarse2fine,1) < eps);
imdl.fwd_model.coarse2fine(:,outside) = [];
imdl.rec_model.coarse2fine(:,outside) = [];
rec_out = [2*outside-1,2*outside];
imdl.rec_model.elems(rec_out,:) = [];

% 创建背景图像
img_1 = mk_image(fmdl, 5);  % 背景电导率 用于做差分的参考背景
vh = fwd_solve(img_1);      % 背景电压
set(0, 'DefaultFigureVisible', 'off');  % 防止弹出过多图片

% 创建文件夹 不加噪声 路径按具体情况调整
% path1 = '../train_data/two_objects_dif/BV/';    %13x16电压矩阵
% path2 = '../train_data/two_objects_dif/Gray/';  %64x64像素矩阵
% path3 = '../train_data/two_objects_dif/DDL/';   %8982x1电导率矩阵 格子数和建模有关

path1 = '../test_data/two_objects_dif/BV/';    %13x16电压矩阵
path2 = '../test_data/two_objects_dif/Gray/';  %64x64像素矩阵
path3 = '../test_data/two_objects_dif/DDL/';   %8982x1电导率矩阵 格子数和建模有关

paths = {path1, path2, path3};  % 将路径存入元胞数组
for k = 1:numel(paths)
    if ~isfolder(paths{k})
        mkdir(paths{k});
    end
end

% 生成数量 可灵活调整 一般训练3000 验证30
num = 30;
for k = 1:num
    % 输出循环数 查看生成数据进度
    disp(k);
    rng('shuffle');
    img_3 = img_1;

    % 随机产生规则图形的中心坐标和半径大小  
    t = randi([0,23]);
    switch t
        case 0
            % 不在同一象限防止相撞
            % 椭圆 1 象限
            x_d = randi([4,10]);
            y_d = randi([4,10]);
            % 圆 2 象限
            x_y = randi([4,10]);
            y_y = randi([-10,-5]);
            % 圆 3 象限
            x_yy = randi([4,10]);
            y_yy = randi([4,10]);
            % 菱形 4 象限
            x_l = randi([4,10]);
            y_l = randi([-10,-5]);

        case 1   

            x_d = randi([4,10]);
            y_d = randi([4,10]);

            x_y = randi([4,10]);
            y_y = randi([-10,-5]);

            x_yy = randi([-10,-5]);
            y_yy = randi([4,10]);

            x_l = randi([-10,-5]);
            y_l = randi([-10,-5]);

        case 2 

            x_d = randi([4,10]);
            y_d = randi([4,10]);

            x_y = randi([4,10]);
            y_y = randi([4,10]);

            x_yy = randi([4,10]);
            y_yy = randi([-10,-5]);
            
            x_l = randi([4,10]);
            y_l = randi([-10,-2]);

        case 3
 
            x_d = randi([4,10]);
            y_d = randi([4,10]);

            x_y = randi([4,10]);
            y_y = randi([4,10]);  

            x_yy = randi([-10,-5]);
            y_yy = randi([4,10]);
            
            x_l = randi([-10,-5]);
            y_l = randi([4,10]);

        case 4

            x_d = randi([4,10]);
            y_d = randi([4,10]);
            
            x_y = randi([-10,-5]);
            y_y = randi([4,10]);

            x_yy = randi([4,10]);
            y_yy = randi([-10,-5]);

            x_l = randi([-10,-5]);
            y_l = randi([-10,-5]);

        case 5

            x_d = randi([4,10]);
            y_d = randi([4,10]);
            
            x_y = randi([-10,-5]);
            y_y = randi([4,10]);

            x_yy = randi([4,10]);
            y_yy = randi([4,10]);
    
            x_l = randi([-10,-5]);
            y_l = randi([4,10]);

        case 6
               
            x_d = randi([-10,-5]);
            y_d = randi([4,10]);
            
            x_y = randi([-10,-5]);
            y_y = randi([-10,-5]);

            x_yy = randi([4,10]);
            y_yy = randi([4,10]);

            x_l = randi([4,10]);
            y_l = randi([-10,-5]);

        case 7   
     
            x_d = randi([-10,-5]);
            y_d = randi([4,10]);

            x_y = randi([-10,-5]);
            y_y = randi([-10,-5]);

            x_yy = randi([-10,-5]);
            y_yy = randi([4,10]);

            x_l = randi([-10,-5]);
            y_l = randi([-10,-5]);

        case 8 

            x_d = randi([-10,-5]);
            y_d = randi([4,10]);
            
            x_y = randi([4,10]);
            y_y = randi([4,10]);  

            x_yy = randi([-10,-5]);
            y_yy = randi([-10,-5]);

            x_l = randi([4,10]);
            y_l = randi([-10,-5]);

        case 9
 
            x_d = randi([-10,-5]);
            y_d = randi([4,10]);

            x_y = randi([4,10]);
            y_y = randi([4,10]);  

            x_yy = randi([-10,-5]);
            y_yy = randi([4,10]);

            x_l = randi([4,10]);
            y_l = randi([4,10]);

        case 10

            x_d = randi([-10,-5]);
            y_d = randi([4,10]);

            x_y = randi([-10,-5]);
            y_y = randi([4,10]);

            x_yy = randi([-10,-5]);
            y_yy = randi([-10,-5]);

            x_l = randi([-10,-2]);
            y_l = randi([-10,-5]);
                
        case 11

            x_d = randi([-10,-5]);
            y_d = randi([4,10]);

            x_y = randi([-10,-5]);
            y_y = randi([4,10]);

            x_yy = randi([4,10]);
            y_yy = randi([4,10]);

            x_l = randi([4,10]);
            y_l = randi([4,10]);

        case 12 

            x_d = randi([-10,-5]);
            y_d = randi([-10,-5]);

            x_y = randi([-10,-5]);
            y_y = randi([-10,-5]);

            x_yy = randi([4,10]);
            y_yy = randi([-10,-5]);

            x_l = randi([4,10]);
            y_l = randi([-10,-5]);

        case 13 

            x_d = randi([-10,-2]);
            y_d = randi([-10,-5]);

            x_y = randi([-10,-5]);
            y_y = randi([-10,-5]);

            x_yy = randi([-10,-5]);
            y_yy = randi([4,10]);

            x_l = randi([-10,-5]);
            y_l = randi([4,10]);

        case 14

            x_d = randi([-10,-5]);
            y_d = randi([-10,-5]);

            x_y = randi([4,10]);
            y_y = randi([-10,-5]);

            x_yy = randi([-10,-5]);
            y_yy = randi([4,10]);

            x_l = randi([4,10]);
            y_l = randi([4,10]);

        case 15 

            x_d = randi([-10,-5]);
            y_d = randi([-10,-5]);

            x_y = randi([4,10]);
            y_y = randi([-10,-5]);

            x_yy = randi([-10,-5]);
            y_yy = randi([-10,-5]);

            x_l = randi([4,10]);
            y_l = randi([-10,-5]);

        case 16 

            x_d = randi([-10,-2]);
            y_d = randi([-10,-5]);

            x_y = randi([-10,-5]);
            y_y = randi([4,10]);

            x_yy = randi([-10,-5]);
            y_yy = randi([-10,-5]);

            x_l = randi([-10,-5]);
            y_l = randi([4,10]);

        case 17 

            x_d = randi([-10,-5]);
            y_d = randi([-10,-5]);

            x_y = randi([-10,-5]);
            y_y = randi([4,10]);

            x_yy = randi([4,10]);
            y_yy = randi([-10,-5]);

            x_l = randi([4,10]);
            y_l = randi([4,10]);

        case 18 

            x_d = randi([4,10]);
            y_d = randi([-10,-5]);

            x_y = randi([-10,-5]);
            y_y = randi([-10,-5]);

            x_yy = randi([4,10]);
            y_yy = randi([-10,-5]);

            x_l = randi([-10,-5]);
            y_l = randi([-10,-5]);

        case 19 

            x_d = randi([4,10]);
            y_d = randi([-10,-5]);

            x_y = randi([-10,-5]);
            y_y = randi([-10,-5]);

            x_yy = randi([4,10]);
            y_yy = randi([4,10]);

            x_l = randi([-10,-5]);
            y_l = randi([4,10]);

        case 20

            x_d = randi([4,10]);
            y_d = randi([-10,-5]);
            
            x_y = randi([4,10]);
            y_y = randi([-10,-5]);

            x_yy = randi([-10,-5]);
            y_yy = randi([-10,-5]);

            x_l = randi([-10,-5]);
            y_l = randi([-10,-5]);

        case 21 

            x_d = randi([4,10]);
            y_d = randi([-10,-5]);

            x_y = randi([4,10]);
            y_y = randi([-10,-5]);

            x_yy = randi([4,10]);
            y_yy = randi([4,10]);

            x_l = randi([4,10]);
            y_l = randi([4,10]);

        case 22 

            x_d = randi([4,10]);
            y_d = randi([-10,-5]);

            x_y = randi([4,10]);
            y_y = randi([4,10]);  

            x_yy = randi([-10,-5]);
            y_yy = randi([-10,-5]);

            x_l = randi([-10,-5]);
            y_l = randi([4,10]);

        case 23 

            x_d = randi([4,10]);
            y_d = randi([-10,-5]);
        
            x_y = randi([4,10]);
            y_y = randi([4,10]);
             
            x_yy = randi([4,10]);
            y_yy = randi([-10,-5]);

            x_l = randi([4,10]);
            y_l = randi([4,10]);

    end

    % 设置半径偏差等参数
    r = randi([4,5]);
    rr = randi([4,5]);
    shaft = randi([2,4]);
    piancha = shaft + randi([1,2]);

    % 生成表达式
    expression_tuoyuan = strcat('((x-',string(x_d),')/',string(shaft),').^2+((y-',string(y_d),')/',string(piancha),').^2<1');
    expression_lingxing = strcat('abs(x-',string(x_l),')+abs(y-',string(y_l),')<',string(r));
    expression_yuan = strcat('(x+',string(x_y),').^2+(y+',string(y_y),').^2<',string(r),'^2');
    expression_yuan2 = strcat('(x+',string(x_yy),').^2+(y+',string(y_yy),').^2<',string(rr),'^2');
    
    % 表达式计算
    select_fcn1 = inline(expression_yuan,'x','y','z');
    select_fcn2 = inline(expression_lingxing,'x','y','z');
    select_fcn3 = inline(expression_yuan2,'x','y','z');
    select_fcn4 = inline(expression_tuoyuan,'x','y','z');

    % 电导率设置 如5是背景电导率 后面是待测物电导率 0代表不加这个待测物 正数代表物体电导率大于背景电导率 负数小于背景电导率 可灵活调整
    % 圆
    % img_3.elem_data = 5+elem_select(img_3.fwd_model, select_fcn2)*0+elem_select(img_3.fwd_model, select_fcn1)*(-2)+elem_select(img_3.fwd_model, select_fcn4)*0;
    % 菱形
    % img_3.elem_data = 5+elem_select(img_3.fwd_model, select_fcn2)*(-2)+elem_select(img_3.fwd_model, select_fcn1)*0+elem_select(img_3.fwd_model, select_fcn4)*0;
    % 两物体(圆+菱形)相同电导率
    % img_3.elem_data = 5+elem_select(img_3.fwd_model, select_fcn2)*(-2)+elem_select(img_3.fwd_model, select_fcn1)*(-2)+elem_select(img_3.fwd_model, select_fcn4)*0;
    % 两物体(圆+菱形)不同电导率
    img_3.elem_data = 5+elem_select(img_3.fwd_model, select_fcn2)*(-4)+elem_select(img_3.fwd_model, select_fcn1)*2+elem_select(img_3.fwd_model, select_fcn4)*0;
    % 三目标(圆+菱形+椭圆)相同电导率
    % img_3.elem_data = 5+elem_select(img_3.fwd_model, select_fcn2)*6+elem_select(img_3.fwd_model, select_fcn1)*6+elem_select(img_3.fwd_model, select_fcn4)*6;
    % 三目标(圆+菱形+椭圆)不同电导率
    % img_3.elem_data = 5+elem_select(img_3.fwd_model, select_fcn2)*10+elem_select(img_3.fwd_model, select_fcn1)*6+elem_select(img_3.fwd_model, select_fcn4)*2;

    % 灰度值图像 
    a = show_slices(img_3);

    % 只展示倒数第二张图 验证生成数据是否正确
    if k == num - 1
        figure('Visible','on');
        % show_fem(img_3); % 网格图
        drawnow;
    end

    % 1.导出灰度矩阵
    fname_gray = fullfile(path2, sprintf('%04d_gray.csv', k));
    writematrix(a, fname_gray);

    % 正模型(获取电压)
    vi = fwd_solve(img_3);

    % 加噪声
    % SNRdb = 30;
    % vi = add_noise(10^(SNRdb/20),vi);

    % 2.导出电压
    % bv = vi.meas - vh.meas;  % 差分电压
    bv = vi.meas;  % 绝对电压
    bv = reshape(bv,[13,16]);
    fname_bv = fullfile(path1, sprintf('%04d_bv.csv', k));
    writematrix(bv, fname_bv);
    
    % 3.导出电导率
    % ddl = img_3.elem_data - img_1.elem_data;  % 差分电导率
    ddl = img_3.elem_data;  % 绝对电导率
    fname_ddl = fullfile(path3, sprintf('%04d_ddl.csv', k));
    writematrix(ddl, fname_ddl);
end