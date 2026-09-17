-- Allow long file paths and commands to wrap; constrain display math to the page.
function Code(el)
  local words = {}
  for word in el.text:gmatch('%S+') do
    table.insert(words, '\\path{' .. word .. '}')
  end
  return pandoc.RawInline('latex', table.concat(words, ' '))
end
function CodeBlock(el)
  return pandoc.RawBlock('latex', '\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\small]\n' .. el.text .. '\n\\end{Verbatim}')
end
function Math(el)
  if el.mathtype == 'DisplayMath' then
    return pandoc.RawInline('latex', '\\[\\begin{adjustbox}{max width=\\linewidth}$\\displaystyle ' .. el.text .. '$\\end{adjustbox}\\]')
  end
end
