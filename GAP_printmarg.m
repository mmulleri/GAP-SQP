function GAP_printmarg(marg,varargin)
  % RI_PRINTMARG prints a reader-friendly description of chosen actions and marginals to
  % the console.
  %
  % RI_PRINTMARG(p) prints a table with the marginals.
  % By default, actions are indexed by the corresponding row in p.
  % This is useful if p spans the entire action grid.
  % RI_PRINTMARG(p,'actionids',a) supplies a list of action indices for each
  % row in p. This is useful if p only contains action within the support.
  % RI_PRINTMARG(p,'actionids',a,'actionlabels',alabels) refers to actions by
  % their full label (alabels) instead of their index.
  %
  % written by Roc Armenter, Michele Muller-Itten and Zachary Stangebye.
  % For further details, see Armenter-et-al, 2019.
  %

  inputs = inputParser;
  addRequired(inputs,'marg',@(x) isnumeric(x) && size(x,2)==1);
  J_curr = size(marg,1); %n of chosen actions
  addParameter(inputs,'actionids',(1:size(marg,1))',@(x) prod(isnumeric(x)) && (prod(size(x)==[J_curr 1])) && prod(mod(x,1)==0));
  addParameter(inputs,'actionlabels',[]);
  parse(inputs,marg,varargin{:});
  actionids = inputs.Results.actionids;
  actionlabels = inputs.Results.actionlabels;
  if length(actionlabels)==0
    actionlabels = (1:1:max(actionids))';
  end
  if (size(actionlabels,1)<max(actionids))
    error("Action index exceeds dimension of the action label list.");
  end

  % remove empty rows
  actionids(marg==0)=[];
  marg(marg==0)=[];

  content = [""];
  for (k=1:size(marg,1))
    content(k,1) = string(strjoin(string(actionlabels(actionids(k),:))));
  end
  fieldwidth = max(max(strlength(content)),6)+1;
  fprintf('%*s | Probability\n',fieldwidth,"Action");
  for (k=1:size(marg,1))
    fprintf('%*s | %1.5g \n',fieldwidth,content(k,1),marg(k));
  end
end
