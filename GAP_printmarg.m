function GAP_printmarg(marg,varargin)
  % RI_PRINTMARG prints a reader-friendly description of chosen actions and marginals to
  % the console.
  %
  % RI_PRINTMARG(p) prints a table with the marginals.
  % By default, actions are indexed by the corresponding row in p.
  % This is useful if p spans the entire action grid.
  % RI_PRINTMARG(p,'actionlabels',alabels) refers to actions by
  % their label (alabels) instead of their index.  
  % RI_PRINTMARG(p,'zero_tol',zerotol) prints only actions with marginal 
  % probability of at least zerotol. Defaults to 0.
  % RI_PRINTMARG(p,'sort',true) sorts by decreasing marginal probability.  
  %
  % written by Roc Armenter, Michele Muller-Itten and Zachary Stangebye.
  % For further details, see Armenter-et-al, 2019.
  %

  inputs = inputParser;
  addRequired(inputs,'marg',@(x) isnumeric(x) && size(x,2)==1);
  J = size(marg,1); %n of actions
  addParameter(inputs,'actionlabels',[],@(x) size(x,1)==J);
  %actionlabels could have multiple columns, which are joined together as
  %a string.
  addParameter(inputs,'zero_tol',0,@(x) isscalar(x) & x>=0);
  addParameter(inputs,'sort',false,@(x) isboolean(x));
  parse(inputs,marg,varargin{:});
  actionlabels = inputs.Results.actionlabels;
  zerotol = inputs.Results.zero_tol;
  s = inputs.Results.sort;
  if length(actionlabels)==0
    actionlabels = (1:N)';
  end

  % remove zero rows
  actionlabels(marg<=zerotol,:)=[];
  marg(marg<=zerotol)=[];
  
  Jprint=size(marg,1);

  content = [""];
  for (k=1:Jprint)
    content(k,1) = strjoin(string(actionlabels(k,:)));
  end
  if (s)
      [~,ord]=sort(marg,'descend');
  else
      ord=1:Jprint;
  end
  fieldwidth = max(max(strlength(content)),6)+1;
  fprintf('%*s | Probability\n',fieldwidth,"Action");
  for (k=1:Jprint)
    fprintf('%*s | %1.5g \n',fieldwidth,content(ord(k),1),marg(ord(k)));
  end
end
