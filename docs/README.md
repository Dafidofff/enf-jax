### Setting up

1. Install node.js and npm
 ```bash
 sudo apt-get install nodejs
 sudo apt-get install npm
 ```

1.5 Install n, a node.js version manager
   ```bash
   sudo npm install -g n
   ```

1.75 update node.js to the latest version
   ```bash
   sudo n latest
   ```

2. Run npm install
 ```bash
 npm install
 ```

3. Install Jekyll and bundler gems
 ```bash
 gem install jekyll bundler
 ```

4. Run bundle install
 ```bash
 bundle install
 ```

### Development and testing
5. Run Jekyll
 ```bash
 bundle exec jekyll serve
 ```

6. Run tailwindcss
 ```bash
 npx tailwindcss -i ./sass/style.scss -o ./assets/css/style.css --watch
 ```
