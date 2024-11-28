if (!window.dash_clientside) {
    window.dash_clientside = {};
}
window.dash_clientside.clientside = {
    make_draggable: function(id) {
        setTimeout(function() {
            var el = document.getElementById(id);
            dragula([el]).on('drop', function(el, target, source, sibling) {
                // 並べ替え後の順序を取得
                const order = Array.from(target.children).map(child => child.querySelector(".card-header").innerText);
                console.log("New Order:", order);
                // 順序をDivに表示
                const outputDiv = document.getElementById('output');
                outputDiv.innerText = `New Order: ${order.join(", ")}`;
            });
        }, 1);
        return window.dash_clientside.no_update;
    }
};
