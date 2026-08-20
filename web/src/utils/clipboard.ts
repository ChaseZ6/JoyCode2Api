export async function writeClipboardText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.cssText = 'position:fixed;left:-9999px;top:0';
  textarea.setAttribute('readonly', '');
  document.body.appendChild(textarea);
  try {
    textarea.select();
    if (!document.execCommand('copy')) {
      throw new Error('Clipboard copy command was rejected');
    }
  } finally {
    document.body.removeChild(textarea);
  }
}
