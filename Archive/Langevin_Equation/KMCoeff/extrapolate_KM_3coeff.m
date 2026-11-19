%-------------------- extrapolate_KM_3coeff.m -----------------------------------
%Function extrapolating the integrals of mean jump values for delays close
%to 0, in order to compute the Kramers-Moyal coefficients. Works only for a
%1D dynamic system.
%Arguments:
%tval: values of the time delays (1D array, size nt)
%KMt: array containing the jump moments for different time delays. 
    % 2D array nt*na with nt the number of time delays and na the number of
    % values for the variable
%tmin: minimal value of time under which the extrapolation is needed.
%init_growth_guess: initial growth rate guess.
function [KM_ext]=extrapolate_KM_3coeff(tval,KMt,tmin,init_growth_guess)
    [nt,na]=size(KMt);
    KM_ext=zeros(nt+1,na);
    
    %index of tmin in the time values array
    t_ind = find(tval>=tmin,1,'first');
    %time values used for the interpolation
    t_rest = tval(t_ind:end);
    t_ext=[0 tval];
    options = optimset('MaxFunEvals',2000*nt);
    
    for ia=1:na
        %data to extrapolate
        to_ext = KMt(t_ind:end,ia);
        %model of extrapolation function
        f_ext = @(a1,a2,a3,x)  a1*exp(a2*x) + a3;

        %d2 distance to optimize
        fopt = @(coeff) sum(   (  to_ext(:) - f_ext(coeff(1),coeff(2),coeff(3),t_rest(:))  ).^2   );

        c_start = [KMt(t_ind,ia)*exp(0.1*tmin),init_growth_guess, 0];
%         disp(    num2str(fopt(c_start))    );
        c = fminsearch(fopt,c_start,options);
        %fitted function
        KM_ext(:,ia) = c(1)*exp(c(2)*t_ext) + c(3);
    end
    
end

