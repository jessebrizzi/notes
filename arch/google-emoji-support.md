# Google emoji support

Use to get the google emojis working in arch linux.

- Install the `noto-fonts-emoji` package.
- Add the following to `/etc/fonts/local.conf`:
```
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>

 <alias>
   <family>sans-serif</family>
   <prefer>
     <family>Your favorite sans-serif font name</family>
     <family>Noto Color Emoji</family>
     <family>Noto Emoji</family>
   </prefer> 
 </alias>

 <alias>
   <family>serif</family>
   <prefer>
     <family>Your favorite serif font name</family>
     <family>Noto Color Emoji</family>
     <family>Noto Emoji</family>
   </prefer>
 </alias>

 <alias>
  <family>monospace</family>
  <prefer>
    <family>Your favorite monospace font name</family>
    <family>Noto Color Emoji</family>
    <family>Noto Emoji</family>
   </prefer>
 </alias>

</fontconfig>
```
- run `fc-cache -rv`
- relauch apps to take affect
